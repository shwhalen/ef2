#!/usr/bin/env python

"""
EnhancerFinder 2.0
Sean Whalen <sean.whalen at gladstone.ucsf.edu>
Pollard Lab, Gladstone Institutes

This code is provided as a reference for the Cell paper "A Chromatin Accessibility Atlas of the Developing Human Telencephalon", and is not intended for use as a tool in this form.  However, it is being developed into a web-based research tool that will be freely accessible in the near future.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import seaborn as sns
import scipy.stats as stats

from chromatics import *
from datetime import datetime
from glob import glob
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, normalize
from tqdm import tqdm
from windborn import *


# feature generation
def preprocess_candidates(atac_fns):
    candidates = pd.concat(
        [read_bed(_, narrowpeak_columns) for _ in atac_fns],
        ignore_index = True
    )

    if args.max_candidate_bp is not None:
        print(f'> trimming candidates to {args.max_candidate_bp}bp for MPRA')
        half_max_candidate_bp = args.max_candidate_bp // 2
        candidates['midpoint'] = candidates.eval('start + peak')
        candidates['start'] = candidates.eval('midpoint - @half_max_candidate_bp')
        candidates['end'] = candidates.eval('midpoint + @half_max_candidate_bp')
        assert candidates.eval('(end - start) == @args.max_candidate_bp').all()

    print('> merging overlapping candidates')
    candidates = bedtools('merge', candidates[bed3_columns])

    if args.max_candidate_bp is not None:
        print(f'> re-trimming candidates to {args.max_candidate_bp}bp')
        candidates['peak'] = candidates.eval('(end - start) / 2').astype(int)
        candidates['midpoint'] = candidates.eval('start + peak')
        candidates['start'] = candidates.eval('midpoint - @half_max_candidate_bp')
        candidates['end'] = candidates.eval('midpoint + @half_max_candidate_bp')
        assert candidates.eval('(end - start) == @args.max_candidate_bp').all()

    print('> dropping duplicates')
    candidates.drop_duplicates(subset = bed3_columns, inplace = True)
    candidates['name'] = concat_coords(candidates)
    candidates.info()

    return candidates


def generate_features(candidates):
    candidates.index = concat_coords(candidates)
    candidates.index.name = 'coord'
    candidate_coords = candidates[bed4_columns]

    kmers = get_kmers(
        get_sequences(candidate_coords),
        (args.kmer_size, args.kmer_size)
    )
    kmers = kmers.loc[:, ~kmers.columns.str.contains('n')]
    kmers = normalize(kmers).astype(np.float32)

    features = [
        get_taka_hits(candidate_coords),
        get_chromatin_loop_hits(candidate_coords),
        get_roadmap_hits(candidate_coords),
        get_conservation(candidate_coords)
    ]
    return pd.concat(features, join = 'inner', axis = 1)


def get_chromatin_loop_hits(query):
    loop_datasets = {
        f'{db_dir}/chromatin_organization/won2016/loops/inter_30-cp-combined_looplist.txt': 'CP Loop Hits',
        f'{db_dir}/chromatin_organization/won2016/loops/inter_30-gz-combined_looplist.txt': 'GZ Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_CH12-LX_HiCCUPS_looplist.txt.gz': 'CH12 Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist.txt.gz': 'GM Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_HeLa-S3_HiCCUPS_looplist.txt.gz': 'HeLa Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_HMEC_HiCCUPS_looplist.txt.gz': 'HMEC Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_HUVEC_HiCCUPS_looplist.txt.gz': 'HUVEC Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_IMR90_HiCCUPS_looplist.txt.gz': 'IMR90 Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_K562_HiCCUPS_looplist.txt.gz': 'K562 Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_KBM7_HiCCUPS_looplist.txt.gz': 'KBM7 Loop Hits',
        f'{db_dir}/chromatin_organization/rao2014/loops/GSE63525_NHEK_HiCCUPS_looplist.txt.gz': 'NHEK Loop Hits'
    }
    loop_hits = Parallel(args.n_jobs)(
        delayed(get_bedpe_hits)(query, fn, feature_name)
        for fn, feature_name in loop_datasets.items()
    )
    return pd.concat(loop_hits, axis = 1)


def get_closest_bed_hits(query, fn, right_names, result_column, feature_name, closest_params = '-D ref -t first'):
    closest_bed_hits = (
        bedtools(
            f'closest {closest_params}',
            query,
            fn,
            right_names = right_names + ['distance'],
            sort = False
        )
        [result_column]
        .rename(feature_name)
    )
    closest_bed_hits.index = query.index

    return closest_bed_hits


def get_conservation(query):
    conservation_datasets = {
        f'{db_dir}/ucsc/phylop/hg19.100way.phyloP100way.bw': 'Conservation (PhyloP 100-Way)',
        f'{db_dir}/ucsc/phastcons/hg19.100way.phastCons.bw': 'Conservation (phastCons 100-Way)'
    }
    signals = Parallel(args.n_jobs)(
        delayed(get_average_bigwig_signal)(query, fn, feature_name)
        for fn, feature_name in conservation_datasets.items()
    )
    peaks = get_bed_counts(
        query,
        'cons/tfbsConsSites-cleaned.bed.gz',
        'Conservation (TFBS)'
    )
    return pd.concat(signals + [peaks], axis = 1)


def get_roadmap_hits(query):
    def format_feature_name(x):
        tissue_id, assay_name = x.split('-')
        return f'{assay_name} ({eid_to_ename[tissue_id]})'

    metadata = (
        pd.read_excel('roadmap/TableS1.xlsx')
        .iloc[2:]
    )
    metadata['Standardized Epigenome name'] = (
        metadata
        ['Standardized Epigenome name']
        .str.replace('_', ' ')
        .str.encode('ascii', 'ignore')
        .str.decode('ascii')
    )
    eid_to_ename = (
        metadata
        .set_index('EID')
        ['Standardized Epigenome name']
        .to_dict()
    )
    excluded_roadmap_eids = {'E028', 'E084', 'E107'}

    fns = glob(f'{db_dir}/roadmap/dnamethylation/hg19/*bigwig')
    eids = [os.path.basename(fn).split('_')[0] for fn in fns]
    feature_names = [f'WGBS ({eid_to_ename[eid]})' for eid in eids]
    wgbs_datasets = dict(zip(fns, feature_names))

    signals = Parallel(args.n_jobs)(
        delayed(get_average_bigwig_signal)(query, fn, feature_name)
        for fn, feature_name in tqdm(wgbs_datasets.items(), 'roadmap wgbs')
    )
    signals = pd.concat(signals, axis = 1)

    print('> intersecting roadmap peaks')
    peaks = (
        bedtools(
            'intersect -sorted -wa -wb',
            query,
            f'{db_dir}/roadmap/narrowPeak/hg19/all_peaks.bed.gz',
            sort = False,
            right_names = ['peak_chrom', 'peak_start',' peak_end', 'feature_name']
        )
        .groupby(bed3_columns + ['feature_name'])
        .size()
        .rename('count')
        .reset_index()
    )
    peaks['coord'] = concat_coords(peaks)
    peaks.drop(bed3_columns, axis = 1, inplace = True)

    print('> pivoting')
    peaks = pd.pivot_table(
        peaks,
        index = 'coord',
        values = 'count',
        columns = 'feature_name',
        aggfunc = sum,
        fill_value = 0
    )

    print('> formatting feature names')
    peaks.columns = (
        peaks
        .columns
        .to_series()
        .apply(format_feature_name)
    )

    print('> reindexing')
    peaks = peaks.reindex(signals.index, fill_value = 0)

    print('> binarizing')
    peaks = (peaks > 0).astype(np.uint8)

    return pd.concat([signals, peaks], axis = 1)


def get_taka_hits(query):
    taka_datasets = json.load(open('taka/filenames.json'))
    peaks = Parallel(args.n_jobs)(
        delayed(get_bed_counts)(query, f'taka/{fn}', feature_name)
        for fn, feature_name in tqdm(taka_datasets.items(), 'taka atac/chip-seq')
    )
    return pd.concat(peaks, axis = 1)


# generate training data and fit/save models
def train():
    def get_vista_hits(coords):
        candidate_intersections = bedtools(
            f'intersect -u -f {args.vista_overlap}',
            candidates,
            coords
        )
        return candidates.eval('name in @candidate_intersections.name')

    def parse_fasta(fn):
        coordinates = []
        sequences = []
        for entry in re.findall(r'>([^>]+)', open(fn).read()):
            lines = entry.split('\n')
            coordinates.append(lines[0])
            sequences.append(''.join(lines[1:]))

        coordinates_sequences = (
            pd.Series(coordinates)
            .str.split('[:-]', expand = True)
        )
        coordinates_sequences.columns = bed3_columns
        coordinates_sequences['sequence'] = sequences

        coordinates_sequences = sort_bed(coordinates_sequences)
        coordinates_sequences.index = concat_coords(coordinates_sequences)
        return coordinates_sequences

    print('[training]')
    os.makedirs(args.output_dir, exist_ok = True)

    if not os.path.exists(f'{args.output_dir}/training.feather'):
        if args.vista_fn is not None:
            atac_fns = args.atac_fns.split(',')
            assert all([os.path.exists(_) for _ in atac_fns])

            print(f'> parsing labels from {args.vista_fn}')
            candidate_re = re.compile(r'>(Human|Mouse)\|(chr[\dXY]{1,2}:\d+-\d+) \| element \d+ \| (positive|negative)')
            outcome_re = re.compile(r'\| +([^\|[]+)\[(\d+)/(\d+)]')

            cns_coords = set()
            other_coords = set()
            negative_coords = set()

            for line in open(args.vista_fn).readlines():
                candidate_match = candidate_re.match(line)
                if (not candidate_match) or candidate_match.group(1) == 'Mouse':
                    continue

                coord = candidate_match.group(2)
                outcome = candidate_match.group(3)
                if outcome == 'negative':
                    negative_coords.add(coord)
                    continue

                cns = False
                for outcome_match in outcome_re.findall(line):
                    tissue = outcome_match[0]
                    percent_validated = int(outcome_match[1]) / int(outcome_match[2])
                    if re.search('brain', tissue) and percent_validated >= 0.5:
                        cns = True
                        break
                if cns:
                    cns_coords.add(coord)
                else:
                    other_coords.add(coord)

            assert len(cns_coords & negative_coords) == 0
            assert len(cns_coords & other_coords) == 0
            assert len(negative_coords & other_coords) == 0

            print('> creating coordinates for each label')
            cns_coords = (
                pd.Series(list(cns_coords))
                .str.split('[:-]', expand = True)
            )
            cns_coords.columns = bed3_columns
            cns_coords['name'] = concat_coords(cns_coords)
            write_bed(cns_coords, f'{args.output_dir}/vista-cns.bed')

            other_coords = (
                pd.Series(list(other_coords))
                .str.split('[:-]', expand = True)
            )
            other_coords.columns = bed3_columns
            other_coords['name'] = concat_coords(other_coords)
            write_bed(other_coords, f'{args.output_dir}/vista-other.bed')

            negative_coords = (
                pd.Series(list(negative_coords))
                .str.split('[:-]', expand = True)
            )
            negative_coords.columns = bed3_columns
            negative_coords['name'] = concat_coords(negative_coords)
            write_bed(negative_coords, f'{args.output_dir}/vista-negatives.bed')

            print('> loading gencode tss annotations')
            tss = pd.read_csv(
                f'{db_dir}/gencode/annotation/hg19/gencode.v19.annotation_capped_sites_nr_with_confidence.gff',
                sep = '\t',
                header = None,
                names = gff_columns
            )
            tss['name'] = concat_coords(tss)
            tss = tss[bed6_columns]

            print('> slopping tss to create promoters')
            promoters = bedtools(
                f'slop -l 1500 -r 500 -s',
                tss
            )

            print('> loading candidates')
            candidates = preprocess_candidates(atac_fns)

            print('\n> removing candidates intersecting promoter regions')
            candidates = bedtools(
                'intersect -wa -v',
                candidates,
                promoters
            )
            candidates.info()

            print('\n> removing candidates intersecting blacklisted regions')
            candidates = bedtools(
                'intersect -wa -v',
                candidates,
                f'{db_dir}/ucsc/blacklist/wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz'
            )
            candidates.info()

            print('\n> adding class labels')
            candidates['negative'] = get_vista_hits(negative_coords)
            print(candidates['negative'].value_counts(), '\n')

            candidates['other'] = get_vista_hits(other_coords)
            print(candidates['other'].value_counts(), '\n')

            candidates['cns'] = get_vista_hits(cns_coords)
            print(candidates['cns'].value_counts(), '\n')

            candidates['unknown'] = candidates.eval('not(cns or other or negative)')
            print(candidates['unknown'].value_counts(), '\n')

            print('> generating features')
            features = generate_features(candidates)

            print('> saving training')
            labels = candidates[label_columns].astype(np.uint8)
            training = (
                pd.concat([labels, features], join = 'inner', axis = 1)
                .reset_index()
            )

            training.to_feather(f'{args.output_dir}/training.feather')
            training, training_features, training_labels, test_features = split_training(f'{args.output_dir}/training.feather')
        elif args.bed_fns is not None:
            bed_fns = args.bed_fns.split(',')
            coords = pd.concat(
                [read_bed(_, names = bed4_columns + ['response']) for _ in bed_fns],
                ignore_index = True,
                sort = False
            )
            coords.index = coords['name']
            coords.index.name = 'coord'

            print('> generating features')
            training_features = generate_features(coords)
            training_labels = coords['response']

            training = (
                pd.concat([training_features, training_labels], join = 'inner', axis = 1)
                .reset_index()
            )
        else:
            raise Exception('Either ATAC-seq peaks or a bed file of genomic coordinates must be provided as input.  Run with -h for help.')
    else:
        training, training_features, training_labels, test_features = split_training(f'{args.output_dir}/training.feather')

    print('> fitting and saving models')
    dummy_model.fit(training_features, training_labels)
    pickle.dump(dummy_model, open(f'{args.output_dir}/dummy_model.pkl', 'wb'))

    ensemble_model.fit(training_features, training_labels)
    pickle.dump(ensemble_model, open(f'{args.output_dir}/ensemble_model.pkl', 'wb'))

    linear_pipeline.fit(training_features, training_labels)
    pickle.dump(linear_pipeline, open(f'{args.output_dir}/linear_pipeline.pkl', 'wb'))


# leave-multiple-chromosomes-out cv
def cross_validate():
    def predict_and_score(estimator):
        def score_predictions(labels, predictions):
            return pd.DataFrame(
                {
                    'f1': f1_score(labels, predictions > args.decision_threshold, average = None),
                    'auroc': roc_auc_score(labels, predictions, average = None),
                    'aupr': average_precision_score(labels, predictions, average = None),
                    'logloss': log_loss(labels.values, predictions)
                },
                index = labels.columns
            )

        predictions = []
        chroms = (
            training
            .index
            .to_series()
            .str.split(':', expand = True)
            .iloc[:, 0]
            .tolist()
        )
        cv = (
            GroupKFold(n_splits = 10)
            .split(
                training_features,
                training_labels,
                chroms
            )
        )

        for train_indices, test_indices in cv:
            estimator.fit(
                training_features.iloc[train_indices],
                training_labels.iloc[train_indices]
            )
            fold_predictions = pd.DataFrame(
                estimator.predict_proba(training_features.iloc[test_indices]),
                columns = training_labels.columns,
                index = test_indices
            )
            predictions.append(fold_predictions)

        predictions = pd.concat(predictions)
        print(estimator)
        print(score_predictions(training_labels.iloc[predictions.index], predictions))

    print('[cross validation]')
    assert os.path.exists(args.input_dir)
    assert os.path.exists(args.output_dir)

    print('> loading training')
    training, training_features, training_labels, test_features = split_training(f'{args.input_dir}/training.feather')

    print('\n> cross-validating baseline')
    dummy_model = pickle.load(open(f'{args.input_dir}/dummy_model.pkl', 'rb'))
    predict_and_score(dummy_model)

    print('\n> cross-validating ensemble')
    ensemble_model = pickle.load(open(f'{args.input_dir}/ensemble_model.pkl', 'rb'))
    predict_and_score(ensemble_model)

    for i, estimator in enumerate(ensemble_model.estimators_):
        current_class_label = training_labels.columns[i]
        importances = (
            pd.Series(
                estimator.feature_importances_,
                index = training_features.columns
            )
            .sort_values(ascending = False)
        )
        importances.to_csv(f'{args.output_dir}/feature_importances-{current_class_label}-ensemble.csv')

        print(f'\n> ensemble {current_class_label}')
        print(importances.head(args.n_top_features))

    print('\n> cross-validating penalized linear model')
    linear_pipeline = pickle.load(open(f'{args.input_dir}/linear_pipeline.pkl', 'rb'))
    predict_and_score(linear_pipeline)

    for i, estimator in enumerate(linear_pipeline.steps[-1][1].estimators_):
        current_class_label = training_labels.columns[i]
        coefs = (
            pd.Series(
                estimator.coef_[0],
                index = training_features.columns.rename('feature'),
                name = 'coefficient'
            )
            .sort_values(ascending = False)
        )
        coefs.to_csv(f'{args.output_dir}/feature_coefs-{current_class_label}-linear.csv')

        print(f'\n> linear {current_class_label}')
        print(coefs.head(args.n_top_features))
        print()
        print(coefs.tail(args.n_top_features))

        # feature correlations
        annotated_colors = (
            [sns.color_palette('Set1')[1]] * args.n_top_features +
            [sns.color_palette('Set1')[0]] * args.n_top_features
        )
        top_feature_names = (
            coefs.head(args.n_top_features).index.tolist() +
            coefs.tail(args.n_top_features).index.tolist()
        )
        cg = sns.clustermap(
            (
                training_features
                [top_feature_names]
                .corr('kendall')
            ),
            cmap = 'PuOr_r',
            vmin = -1,
            vmax = 1,
            col_colors = annotated_colors
        )
        plt.setp(cg.ax_heatmap.get_yticklabels(), rotation = 0)
        plt.savefig(
            f'{args.output_dir}/top_feature_corr-{current_class_label}-linear.pdf',
            bbox_inches = 'tight'
        )

        # feature coefficients
        plt.figure(figsize = (4, 12))
        top_features = pd.concat([
            coefs.head(3 * args.n_top_features),
            coefs.tail(3 * args.n_top_features)
        ])
        sns.stripplot(
            x = 'coefficient',
            y = 'feature',
            data = top_features[top_features != 0].reset_index(),
            palette = 'Blues_r',
            linewidth = 1
        )
        plt.axvline(0, c = 'silver')
        plt.gca().xaxis.set_label_position('top')
        plt.gca().tick_params(labelbottom = True, labeltop = True)
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.savefig(
            f'{args.output_dir}/top_feature_coefs-{current_class_label}-linear.pdf',
            bbox_inches = 'tight'
        )


def predict():
    def get_predictions(estimator, estimator_type):
        predictions = pd.DataFrame(
            estimator.predict_proba(test_features[common_features])[:, 1],
            columns = [f'{estimator_type}_prediction'],
            index = test_features.index
        )
        predictions[f'{estimator_type}_prediction'] = predictions[f'{estimator_type}_prediction'].round(3)
        return predictions

    def get_annotations(candidate_coords):
        def get_domain_genes(domains_fn, cell_type):
            domains = pd.read_csv(
                domains_fn,
                sep = '\t',
                header = 0,
                usecols = range(3),
                names = ['domain_chrom', 'domain_start', 'domain_end']
            )
            domains = bedtools('merge', domains)

            domain_genes = (
                bedtools(
                    'intersect -F 1.0 -wa -wb',
                    domains,
                    genes,
                    sort = False
                )
                .groupby(domains.columns.tolist())
                .apply(
                    lambda x: ','.join(
                        x['gene_name']
                        .sort_values()
                    )
                )
                .rename(f'{cell_type}_intra_domain_genes')
                .reset_index()
            )

            candidate_domain_genes = bedtools(
                'intersect -wa -wb -f 0.9 -loj',
                candidate_coords,
                domain_genes,
                sort = False
            )
            candidate_domain_genes.index = candidate_coords.index
            return candidate_domain_genes[f'{cell_type}_intra_domain_genes'].str.replace('.', '', regex = False)

        print('> getting protein coding gene coordinates')
        genes = pd.read_csv(
            f'{db_dir}/biomart/hg19_pc_genes.tsv',
            sep = '\t',
            header = 0,
            names = ['gene_id', 'gene_name', 'gene_chrom', 'gene_start', 'gene_end']
        )
        genes = genes[~genes['gene_name'].str.startswith('RP11')]
        genes = reorder_columns(genes, ['gene_chrom', 'gene_start', 'gene_end'])
        genes['gene_chrom'] = 'chr' + genes['gene_chrom'].astype(str).replace('MT', 'M')
        genes = sort_bed(genes)

        print('> getting exon/intron/intergenic overlaps')
        exon_counts = get_bed_counts(
            candidate_coords,
            f'{db_dir}/gencode/annotation/hg19/exon.bed.gz',
            'n_exons',
            binarize = False
        )

        intron_counts = get_bed_counts(
            candidate_coords,
            f'{db_dir}/gencode/annotation/hg19/intron.bed.gz',
            'n_introns',
            binarize = False
        )

        intergenic_counts = get_bed_counts(
            candidate_coords,
            f'{db_dir}/gencode/annotation/hg19/intergenic.bed.gz',
            'n_intergenic',
            binarize = False
        )

        print('> getting distances to vista enhancers')
        closest_cns_vista_enhancer = get_closest_bed_hits(
            candidate_coords,
            f'{args.input_dir}/vista-cns.bed',
            ['vista_chrom', 'vista_start', 'vista_end', 'vista_name'],
            'distance',
            'distance_to_closest_cns_vista_enhancer'
        )

        closest_non_cns_vista_enhancer = get_closest_bed_hits(
            candidate_coords,
            f'{args.input_dir}/vista-other.bed',
            ['vista_chrom', 'vista_start', 'vista_end', 'vista_name'],
            'distance',
            'distance_to_closest_non_cns_vista_enhancer'
        )

        print('> getting closest protein coding genes')
        closest_upstream_pc_genes = get_closest_bed_hits(
            candidate_coords,
            genes,
            genes.columns.tolist(),
            'gene_name',
            'closest_upstream_pc_gene',
            '-D ref -t first -fu'
        )

        closest_downstream_pc_genes = get_closest_bed_hits(
            candidate_coords,
            genes,
            genes.columns.tolist(),
            'gene_name',
            'closest_downstream_pc_gene',
            '-D ref -t first -fd'
        )

        print('> getting pfc k27ac overlaps')
        pfc_dl_k27ac_hits = get_bed_counts(
            candidate_coords,
            '../doca/peaks/22gw_1DL_ak27_S10_L002_R1_001.PE2SE.nodup_pooled.tagAlign_x_22gw_input_S17_L002_R1_001.PE2SE.nodup.tagAlign.pval0.01.500K.filt.narrowPeak.gz',
            'overlaps_pfc_dl_k27ac_peak'
        )

        pfc_ul_k27ac_hits = get_bed_counts(
            candidate_coords,
            '../doca/peaks/22gw_1UL_ak27_S9_L002_R1_001.PE2SE.nodup_pooled.tagAlign_x_22gw_input_S17_L002_R1_001.PE2SE.nodup.tagAlign.pval0.01.500K.filt.narrowPeak.gz',
            'overlaps_pfc_ul_k27ac_peak'
        )

        print('> getting overlapping region atac peaks')
        emp_regions = ['cge', 'lge', 'mge', 'motor', 'parietal', 'pfc', 'somato', 'temporal', 'v1']
        all_region_peak_hits = []
        for region in tqdm(emp_regions):
            region_fns = glob(f'../doca/peaks/*{region}*narrowPeak.gz')
            region_peaks = pd.concat(
                [read_bed(_, narrowpeak_columns) for _ in region_fns],
                ignore_index = True
            )
            region_peak_hits = get_bed_counts(
                candidate_coords,
                region_peaks,
                f'overlaps_{region}_peak'
            )
            all_region_peak_hits.append(region_peak_hits)
        region_peak_hits = pd.concat(all_region_peak_hits, axis = 1)

        print('> getting intra-domain genes')
        cp_domain_genes = get_domain_genes(
            f'{db_dir}/chromatin_organization/won2016/domains/inter_30-cp-combined_domainlist_10000_blocks',
            'cortical_plate'
        )

        gz_domain_genes = get_domain_genes(
            f'{db_dir}/chromatin_organization/won2016/domains/inter_30-gz-combined_domainlist_10000_blocks',
            'germinal_zone'
        )

        print('> getting intra-domain genes with high single cell expression')
        sc_expression = (
            pd.read_feather('../doca/tomasz/single_cell_cpm.feather')
            .set_index('gene_name')
            .mean(axis = 1)
        )
        top_sc_genes = set(sc_expression[sc_expression > sc_expression.quantile(0.9)].index)
        top_sc_genes_in_cp_domain = (
            cp_domain_genes
            .apply(lambda x: [_ for _ in x.split(',') if _ in top_sc_genes])
            .apply(lambda x: ','.join(x))
            .rename('top_sc_genes_in_cp_domain')
        )
        top_sc_genes_in_gz_domain = (
            gz_domain_genes
            .apply(lambda x: [_ for _ in x.split(',') if _ in top_sc_genes])
            .apply(lambda x: ','.join(x))
            .rename('top_sc_genes_in_gz_domain')
        )

        if args.bed_fns is None:
            conservation = (
                pd.concat([training_features, test_features])
                [['Conservation (PhyloP 100-Way)', 'Conservation (phastCons 100-Way)']]
            )
        else:
            conservation = get_conservation(candidate_coords)
        conservation = (
            conservation
            .rename(
                columns = {
                    'Conservation (PhyloP 100-Way)': 'conservation_phylop_100way',
                    'Conservation (phastCons 100-Way)': 'conservation_phastcons_100way'
                }
            )
            .round(3)
        )

        return (
            pd.concat(
                [
                    candidate_coords,
                    conservation,
                    exon_counts,
                    intron_counts,
                    intergenic_counts,
                    closest_cns_vista_enhancer,
                    closest_non_cns_vista_enhancer,
                    closest_upstream_pc_genes,
                    closest_downstream_pc_genes,
                    pfc_dl_k27ac_hits,
                    pfc_ul_k27ac_hits,
                    region_peak_hits,
                    cp_domain_genes,
                    top_sc_genes_in_cp_domain,
                    gz_domain_genes,
                    top_sc_genes_in_gz_domain
                ],
                axis = 1,
                join = 'inner',
                sort = False
            )
            .drop(bed3_columns, axis = 1)
        )

    print('[predicting]')
    assert os.path.exists(args.input_dir)
    assert os.path.exists(args.output_dir)

    print('> loading training')
    training, training_features, training_labels, test_features = split_training(f'{args.input_dir}/training.feather')

    if args.bed_fns is not None:
        bed_fns = args.bed_fns.split(',')
        assert all([os.path.exists(_) for _ in bed_fns])

        print('> loading candidates')
        candidates = preprocess_candidates(bed_fns)

        print('> generating features')
        test_features = generate_features(candidates)

    print('> annotating candidates')
    test_candidate_coords = (
        test_features
        .index
        .to_series()
        .str.split('[:-]', expand = True)
    )
    test_candidate_coords.columns = bed3_columns
    test_candidate_annotations = get_annotations(test_candidate_coords)

    print('> fitting models')
    common_features = test_features.columns.intersection(training_features.columns)
    print(f'> features: {len(common_features)} shared, {training_features.shape[1]} training, {test_features.shape[1]} test')

    # load from saved model to ensure parameters won't change, but re-fit to common set of features
    ensemble_model = pickle.load(open(f'{args.input_dir}/ensemble_model.pkl', 'rb'))
    ensemble_model.fit(training_features[common_features], training_labels['cns'])

    linear_pipeline = pickle.load(open(f'{args.input_dir}/linear_pipeline.pkl', 'rb'))
    linear_pipeline.fit(training_features[common_features], training_labels['cns'])

    print('> predicting')
    linear_predictions = get_predictions(linear_pipeline, 'linear')
    ensemble_predictions = get_predictions(ensemble_model, 'ensemble')
    annotated_predictions = pd.concat(
        [test_candidate_annotations, linear_predictions, ensemble_predictions],
        axis = 1,
        sort = False
    )

    (
        annotated_predictions
        .sort_values('ensemble_prediction', ascending = False)
        .to_csv(f'{args.output_dir}/predictions-cns.csv', index_label = 'coord')
    )

    plot_correlation(
        annotated_predictions,
        'linear_prediction',
        'ensemble_prediction',
         'Linear Prediction',
         'Ensemble Prediction',
         f'{args.output_dir}/predictions-correlation.png'
    )

    training_candidate_coords = (
        training_features
        .index
        .to_series()
        .str.split('[:-]', expand = True)
    )
    training_candidate_coords.columns = bed3_columns
    training_label_annotations = training_labels.rename(columns = lambda x: 'label_' + x)
    (
        pd.concat(
            [
                get_annotations(training_candidate_coords),
                training_label_annotations
            ],
            axis = 1,
            join = 'inner'
        )
        .sort_values(training_label_annotations.columns.tolist(), ascending = False)
        .to_csv(f'{args.output_dir}/annotations-training.csv', index_label = 'coord')
    )


def split_training(training_fn):
    # genome-wide candidates
    all_training = (
        pd.read_feather(training_fn)
        .set_index('coord')
    )

    # select candidates with a known vista label for training set
    training = all_training.query('unknown == 0')
    training_features = training.drop(label_columns, axis = 1)
    training_labels = (
        training
        [label_columns[:3]]
        .astype(int)
    )

    # select candidates without a known vista label for test set (prediction)
    test_features = (
        all_training
        .query('unknown == 1')
        .drop(label_columns, axis = 1)
    )

    print(f'training dim: {all_training.shape}')
    print('      labels:')
    print(all_training[label_columns].sum(axis = 0))

    print('multi-label samples:')
    print(training_labels[training_labels.sum(axis = 1) > 1])

    return training, training_features, training_labels, test_features


sns.set(style = 'white')
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

label_columns = ['cns', 'other', 'negative', 'unknown']
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest = 'command')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('-a', '--atac_fns')
train_parser.add_argument('-b', '--bed_fns')
train_parser.add_argument('-v', '--vista_fn')
train_parser.add_argument('-vo', '--vista_overlap', type = float, default = 0.51)
train_parser.add_argument('-k', '--kmer_size', type = int, default = 5)
train_parser.add_argument('-m', '--max_candidate_bp', type = int)
train_parser.add_argument('-o', '--output_dir')
train_parser.add_argument('-j', '--n_jobs', type = int, default = -2)

cv_parser = subparsers.add_parser('cv')
cv_parser.add_argument('-t', '--decision_threshold', type = float, default = 0.5)
cv_parser.add_argument('-n', '--n_top_features', type = int, default = 8)
cv_parser.add_argument('-i', '--input_dir')
cv_parser.add_argument('-o', '--output_dir')
cv_parser.add_argument('-j', '--n_jobs', type = int, default = -2)

predict_parser = subparsers.add_parser('predict')
predict_parser.add_argument('-b', '--bed_fns')
predict_parser.add_argument('-vo', '--vista_overlap', type = float, default = 0.51)
predict_parser.add_argument('-k', '--kmer_size', type = int, default = 5)
predict_parser.add_argument('-m', '--max_candidate_bp', type = int)
predict_parser.add_argument('-i', '--input_dir')
predict_parser.add_argument('-o', '--output_dir')
predict_parser.add_argument('-j', '--n_jobs', type = int, default = -2)

print(r'''
        _____________     _______
  _____/ ____\_____  \    \   _  \
_/ __ \   __\ /  ____/    /  /_\  \
\  ___/|  |  /       \    \  \_/   \
 \___  >__|  \_______ \ /\ \_____  /
     \/              \/ \/       \/
''')

args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = args.input_dir

dummy_model = OneVsRestClassifier(
    DummyClassifier(random_state = 0)
)

# lasso (l1) = alpha * sum(abs(coefs)), ridge (l2) = alpha * sum(coefs**2), larger alpha = more sparse
linear_model = OneVsRestClassifier(
    SGDClassifier(
        loss = 'log',
        penalty = 'elasticnet',
        max_iter = 1000,
        tol = 1e-3,
        class_weight = 'balanced',
        n_jobs = args.n_jobs,
        random_state = 0
    )
)
linear_pipeline = make_pipeline(
    Normalizer(norm = 'l2'),
    linear_model
)

ensemble_model = OneVsRestClassifier(
    ExtraTreesClassifier(
        n_estimators = 2000,
        max_features = 'sqrt',
        n_jobs = args.n_jobs,
        random_state = 0
    )
)

if args.command == 'train':
    train()
elif args.command == 'cv':
    cross_validate()
else:
    predict()
