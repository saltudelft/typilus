#!/usr/bin/env python
"""
Usage:
    test.py [options] PREDICTIONS_JSONL_GZ TYPE_LATTICE_PATH ALIAS_METADATA_PATH

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""

from collections import defaultdict

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from sklearn.metrics import classification_report

from typilus.model.typelattice import TypeLattice
from typilus.model.utils import ignore_type_annotation
from tqdm import tqdm
import json
import numpy as np

def compute(predictions_path: RichPath, type_lattice_path: RichPath, alias_metadata_path: RichPath,
            json_preds: str, top_n: int = 10):

    with open(json_preds) as f:
        json_data = json.load(f)['per_type_stats']

    type_lattice = TypeLattice(type_lattice_path, 'typing.Any', alias_metadata_path)
    data = predictions_path.read_as_jsonl()

    total_per_kind = defaultdict(int)
    correct_per_kind = defaultdict(int)
    up_to_parameteric_per_kind = defaultdict(int)
    # type_consistency_per_kind = defaultdict(int)
    # total_per_kind_for_consistency = defaultdict(int)

    ubiq_t = {'str', 'int', 'list', 'bool', 'float', 'typing.Text', 'typing.List', 'typing.List[typing.Any]', 'typing.list'}
    common_tr = 100

    corr_exact_per_kind = defaultdict(lambda: defaultdict(int))
    corr_param_per_kind = defaultdict(lambda: defaultdict(int))
    mrr_exact_per_kind = defaultdict(lambda: defaultdict(list))
    mrr_param_per_kind = defaultdict(lambda: defaultdict(list))
    corr_exact = defaultdict(int)
    corr_param = defaultdict(int)

    # For recall and precision calculation
    pred_annotation = []
    true_annotation = []
    pred_per_type = defaultdict(list)
    true_per_type = defaultdict(list)

    mrr_all = []
    mrr_exact_ubiq = []
    mrr_exact_comm = []
    mrr_exact_rare = []

    mrr_param_all = []
    mrr_param_comm = []
    mrr_param_rare = []

    for prediction in data:
        annotation_type = prediction['annotation_type']
        original_annotation = prediction['original_annotation']
        if ignore_type_annotation(original_annotation) or annotation_type == 'imported':
            continue
        
        true_annotation.append(original_annotation)
        true_per_type[annotation_type].append(original_annotation)
        total_per_kind[annotation_type] += 1
        #top_predicted = prediction['predicted_annotation_logprob_dist'][0][0]
        #print(prediction['predicted_annotation_logprob_dist'])
        is_exact_match = False
        is_accurate_utpt = False

        for i, (t, s) in enumerate(prediction['predicted_annotation_logprob_dist'][:top_n]):
            is_exact_match = type_lattice.are_same_type(original_annotation, t)
            r = 1/(i+1)
            if is_exact_match:
                correct_per_kind[annotation_type] += 1
                corr_exact['all'] += 1
                mrr_all.append(r)
                mrr_exact_per_kind[annotation_type]['mrr_all'].append(r)
                if original_annotation in ubiq_t:
                    #print(original_annotation)
                    corr_exact_per_kind[annotation_type]['corr_ubiq'] += 1
                    corr_exact['corr_ubiq'] += 1
                    mrr_exact_per_kind[annotation_type]['mrr_ubiq'].append(r)
                    mrr_exact_ubiq.append(r)
                elif json_data[original_annotation]['count'] >= common_tr:
                    #print("C", original_annotation)
                    corr_exact_per_kind[annotation_type]['corr_common'] += 1
                    corr_exact['corr_common'] += 1
                    mrr_exact_per_kind[annotation_type]['mrr_comm'].append(r)
                    mrr_exact_comm.append(r)
                else:
                    corr_exact_per_kind[annotation_type]['corr_rare'] += 1
                    corr_exact['corr_rare'] += 1
                    mrr_exact_per_kind[annotation_type]['mrr_rare'].append(r)
                    mrr_exact_rare.append(r)
                
                pred_annotation.append(t)
                pred_per_type[annotation_type].append(t)
                break
                
        for i, (t, s) in enumerate(prediction['predicted_annotation_logprob_dist'][:top_n]):
            is_accurate_utpt = type_lattice.are_same_type(original_annotation.split("[")[0], t.split("[")[0])
            r = 1/(i+1)
            if is_accurate_utpt:
                up_to_parameteric_per_kind[annotation_type] += 1
                corr_param['all'] += 1
                mrr_param_all.append(r)
                mrr_param_per_kind[annotation_type]['mrr_all'].append(r)
                if original_annotation in ubiq_t:
                    corr_param_per_kind[annotation_type]['corr_ubiq'] += 1
                    corr_param['corr_ubiq'] += 1
                    
                elif json_data[original_annotation]['count'] >= common_tr:
                    corr_param_per_kind[annotation_type]['corr_common'] += 1
                    corr_param['corr_common'] += 1
                    mrr_param_per_kind[annotation_type]['mrr_comm'].append(r)
                    mrr_param_comm.append(r)
                else:
                    corr_param_per_kind[annotation_type]['corr_rare'] += 1
                    corr_param['corr_rare'] += 1
                    mrr_param_per_kind[annotation_type]['mrr_rare'].append(r)
                    mrr_param_rare.append(r)
                break

        if not is_exact_match:
            mrr_all.append(0)
            mrr_exact_per_kind[annotation_type]['mrr_all'].append(0)
            if original_annotation in ubiq_t:
                corr_exact_per_kind[annotation_type]['incorr_ubiq'] += 1
                corr_exact['incorr_ubiq'] += 1
                mrr_exact_per_kind[annotation_type]['mrr_ubiq'].append(0)
                mrr_exact_ubiq.append(0)
            elif json_data[original_annotation]['count'] >= common_tr:
                corr_exact_per_kind[annotation_type]['incorr_common'] += 1
                corr_exact['incorr_common'] += 1
                mrr_exact_per_kind[annotation_type]['mrr_comm'].append(0)
                mrr_exact_comm.append(0)
            else:
                corr_exact_per_kind[annotation_type]['incorr_rare'] += 1
                corr_exact['incorr_rare'] += 1
                mrr_exact_per_kind[annotation_type]['mrr_rare'].append(0)
                mrr_exact_rare.append(0)
            pred_annotation.append(prediction['predicted_annotation_logprob_dist'][0][0])
            pred_per_type[annotation_type].append(prediction['predicted_annotation_logprob_dist'][0][0])

        if not is_accurate_utpt:
            mrr_param_all.append(0)
            mrr_param_per_kind[annotation_type]['mrr_all'].append(0)
            if original_annotation in ubiq_t:
                corr_param_per_kind[annotation_type]['incorr_ubiq'] += 1
                corr_param['incorr_ubiq'] += 1
            if json_data[original_annotation]['count'] >= common_tr:
                corr_param_per_kind[annotation_type]['incorr_common'] += 1
                corr_param['incorr_common'] += 1
                mrr_param_per_kind[annotation_type]['mrr_comm'].append(0)
                mrr_param_comm.append(0)
            else:
                corr_param_per_kind[annotation_type]['incorr_rare'] += 1
                corr_param['incorr_rare'] += 1
                mrr_param_per_kind[annotation_type]['mrr_rare'].append(0)
                mrr_param_rare.append(0)
        # if is_exact_match:
        #     type_consistency_per_kind[annotation_type] += 1
        #     total_per_kind_for_consistency[annotation_type] += 1
        # elif original_annotation in type_lattice and top_predicted in type_lattice:
        #     # Type Consistency
        #     ground_truth_node_idx = type_lattice.id_of(original_annotation)
        #     predicted_node_idx = type_lattice.id_of(top_predicted)

        #     intersection_nodes_idx = type_lattice.intersect(ground_truth_node_idx, predicted_node_idx)
        #     is_ground_subtype_of_predicted = ground_truth_node_idx in intersection_nodes_idx
        #     total_per_kind_for_consistency[annotation_type] += 1
        #     if is_ground_subtype_of_predicted:
        #         type_consistency_per_kind[annotation_type] += 1

    print('== Exact Match')
    for annot_type in total_per_kind:
        try:
            print(f'{annot_type}: {correct_per_kind[annot_type] / total_per_kind[annot_type] * 100.0 :.1f} ({correct_per_kind[annot_type]}/{total_per_kind[annot_type]})')
            print(f"Ubiq - {annot_type}: {corr_exact_per_kind[annot_type]['corr_ubiq'] / (corr_exact_per_kind[annot_type]['corr_ubiq'] + corr_exact_per_kind[annot_type]['incorr_ubiq']) * 100.0 :.1f}")
            print(f"Common - {annot_type}: {corr_exact_per_kind[annot_type]['corr_common'] / (corr_exact_per_kind[annot_type]['corr_common'] + corr_exact_per_kind[annot_type]['incorr_common']) * 100.0 :.1f}")
            print(f"Rare - {annot_type}: {corr_exact_per_kind[annot_type]['corr_rare'] / (corr_exact_per_kind[annot_type]['corr_rare'] + corr_exact_per_kind[annot_type]['incorr_rare']) * 100.0 :.1f}")
            print(f"MRR - {annot_type} - ALL: {np.mean(mrr_exact_per_kind[annot_type]['mrr_all'])*100:.1f} ubiq: {np.mean(mrr_exact_per_kind[annot_type]['mrr_ubiq'])*100:.1f} comm: {np.mean(mrr_exact_per_kind[annot_type]['mrr_comm'])*100:.1f} rare: {np.mean(mrr_exact_per_kind[annot_type]['mrr_rare'])*100:.1f}")
            
            #r = classification_report(true_per_type[annot_type], pred_per_type[annot_type], output_dict=True)
            #print(f"{annot_type}: F1: {r['weighted avg']['f1-score'] * 100:.1f} R: {r['weighted avg']['recall'] * 100:.1f} P: {r['weighted avg']['precision'] * 100:.1f}")
            print("******************************")
        except ZeroDivisionError:
            pass
    print('== Up to Parametric')
    for annot_type in total_per_kind:
        try:
            print(f'{annot_type}: {up_to_parameteric_per_kind[annot_type] / total_per_kind[annot_type] * 100.0 :.1f} ({up_to_parameteric_per_kind[annot_type]}/{total_per_kind[annot_type]})')
            print(f"Common - {annot_type}: {corr_param_per_kind[annot_type]['corr_common'] / (corr_param_per_kind[annot_type]['corr_common'] + corr_param_per_kind[annot_type]['incorr_common']) * 100.0 :.1f}")
            print(f"Rare - {annot_type}: {corr_param_per_kind[annot_type]['corr_rare'] / (corr_param_per_kind[annot_type]['corr_rare'] + corr_param_per_kind[annot_type]['incorr_rare']) * 100.0 :.1f}")
            print(f"MRR - {annot_type} - ALL: {np.mean(mrr_param_per_kind[annot_type]['mrr_all'])*100:.1f} comm: {np.mean(mrr_param_per_kind[annot_type]['mrr_comm'])*100:.1f} rare: {np.mean(mrr_param_per_kind[annot_type]['mrr_rare'])*100:.1f}")
            print("******************************")
        except ZeroDivisionError:
            pass

    # r = classification_report(true_annotation, pred_annotation, output_dict=True)
    # print("Precision: %.1f" % (r['weighted avg']['precision'] * 100))
    # print("Recall: %.1f" % (r['weighted avg']['recall'] * 100))
    # print("F1-score: %.1f" % (r['weighted avg']['f1-score'] * 100))
    print("******************************")
    print(f"Exact - All: {corr_exact['all']/len(true_annotation)*100.0:.1f} ubiq: {corr_exact['corr_ubiq'] / (corr_exact['corr_ubiq'] + corr_exact['incorr_ubiq'])*100.0:.1f} common: {corr_exact['corr_common'] / (corr_exact['corr_common'] + corr_exact['incorr_common'])*100.0:.1f} rare: {corr_exact['corr_rare'] / (corr_exact['corr_rare'] + corr_exact['incorr_rare'])*100.0:.1f}")
    print(f"Parameteric - All: {corr_param['all']/len(true_annotation)*100.0:.1f} common: {corr_param['corr_common'] / (corr_param['corr_common'] + corr_param['incorr_common'])*100.0:.1f} rare: {corr_param['corr_rare'] / (corr_param['corr_rare'] + corr_param['incorr_rare'])*100.0:.1f}")
    print(f"MRR Exact All: {np.mean(mrr_all)*100:.1f} MRR ubiq: {np.mean(mrr_exact_ubiq)*100:.1f} MRR comm: {np.mean(mrr_exact_comm)*100:.1f} MRR rare: {np.mean(mrr_exact_rare)*100:.1f} ")
    print(f"MRR Param All: {np.mean(mrr_param_all)*100:.1f} MRR comm: {np.mean(mrr_param_comm)*100:.1f} MRR rare: {np.mean(mrr_param_rare)*100:.1f} ")

    return np.mean(mrr_all)*100
    # print('== Consistency')
    # for annot_type in total_per_kind:
    #     print(f'{annot_type}: {type_consistency_per_kind[annot_type] / total_per_kind_for_consistency[annot_type] :%} ({type_consistency_per_kind[annot_type]}/{total_per_kind_for_consistency[annot_type]})')



def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    predictions_path = RichPath.create(arguments['PREDICTIONS_JSONL_GZ'], azure_info_path)
    type_lattice_path = RichPath.create(arguments['TYPE_LATTICE_PATH'], azure_info_path)
    alias_metadata_path = RichPath.create(arguments['ALIAS_METADATA_PATH'], azure_info_path)
    compute(predictions_path, type_lattice_path, alias_metadata_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
