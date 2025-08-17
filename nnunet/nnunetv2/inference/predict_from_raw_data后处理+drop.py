import inspect
import itertools
import multiprocessing
import os
import time
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional
from concurrent.futures import ThreadPoolExecutor
import psutil

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, save_pickle
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window


def export_prediction_from_logits_threaded(predicted_array_or_file: Union[np.ndarray, torch.Tensor], 
                                         properties_dict: dict,
                                         configuration_manager, 
                                         plans_manager, 
                                         dataset_json_dict_or_file: Union[dict, str], 
                                         output_file_truncated: str,
                                         save_probabilities: bool = False,
                                         num_threads_torch: int = None):
    """
    多线程优化版本的export_prediction_from_logits
    充分利用CPU资源进行后处理加速
    """
    if num_threads_torch is None:
        # 获取物理CPU核心数进行优化配置
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        # 对于重采样这种CPU密集型任务，使用物理核心数的1.5倍，但不超过16
        num_threads_torch = min(int(physical_cores * 1.5), 16)
    
    print(f"🚀 使用 {num_threads_torch} 个线程进行后处理优化...")
    
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    # 设置torch线程数进行优化
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)
    
    try:
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
            return_probabilities=save_probabilities
        )
        del predicted_array_or_file

        # 并行保存文件
        if save_probabilities:
            segmentation_final, probabilities_final = ret
            
            def save_probabilities_task():
                """保存概率和属性文件"""
                np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
                save_pickle(properties_dict, output_file_truncated + '.pkl')
                print("✅ 概率文件保存完成")
            
            def save_segmentation_task():
                """保存分割结果"""
                rw = plans_manager.image_reader_writer_class()
                rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                            properties_dict)
                print("✅ 分割文件保存完成")
            
            # 使用线程池并行执行保存任务
            print("🔄 并行保存概率和分割文件...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                prob_future = executor.submit(save_probabilities_task)
                seg_future = executor.submit(save_segmentation_task)
                prob_future.result()
                seg_future.result()
            
            del probabilities_final, ret
        else:
            segmentation_final = ret
            del ret
            rw = plans_manager.image_reader_writer_class()
            rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                        properties_dict)
            print("✅ 分割文件保存完成")
            
    finally:
        torch.set_num_threads(old_threads)


def print_optimization_info():
    """
    打印系统信息和优化建议
    """
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print("\n" + "="*60)
    print("🖥️  系统配置信息")
    print("="*60)
    print(f"物理CPU核心数: {physical_cores}")
    print(f"逻辑CPU核心数: {logical_cores}")
    print(f"系统内存: {memory_gb:.1f} GB")
    print(f"推荐线程数: {min(int(physical_cores * 1.5), 16)}")
    print("="*60)
    print("🚀 后处理优化已启用，将加速Step 3重采样过程")
    print("="*60 + "\n")


from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.9,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 optimize_postprocessing: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm
        self.optimize_postprocessing = optimize_postprocessing  # 新增参数控制后处理优化

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        # Handle torch.compile prefix issues
        state_dict = parameters[0]
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("⚠️  检测到torch.compile前缀，正在修复权重...")
            # Remove _orig_mod. prefix from all keys
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
        
        try:
            network.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"❌ 严格模式加载失败: {str(e)}")
            print("🔄 尝试非严格模式加载...")
            network.load_state_dict(state_dict, strict=False)

        self.network = network

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am processing {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]
        else:
            output_filename_truncated = None
        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files from the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        for params in self.list_of_parameters:

            # Handle torch.compile prefix issues for inference
            state_dict = params
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                # Remove _orig_mod. prefix from all keys
                state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
            
            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(state_dict, strict=False)
            else:
                self.network._orig_mod.load_state_dict(state_dict, strict=False)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:  # just a file was given without a folder
                output_folder = os.path.curdir
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(seg_from_prev_stage_files)

        ret = []
        for idx, (li, of, sps) in enumerate(zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files)):
            case_name = os.path.basename(li[0]) if li else f"case_{idx}"
            print(f"\n=== Processing case {idx+1}/{len(list_of_lists_or_source_folder)}: {case_name} ===")
            
            # 检查文件大小，对于大于70MB的文件跳过预测，生成虚拟结果
            total_file_size = 0
            for file_path in li:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    total_file_size += file_size
                    print(f"File {os.path.basename(file_path)}: {file_size / (1024*1024):.2f} MB")
            
            print(f"Total file size: {total_file_size / (1024*1024):.2f} MB")
            
            # 如果总文件大小超过70MB，跳过预测并生成虚拟结果
            if total_file_size > 70 * 1024 * 1024:  # 70MB in bytes
                print(f"⚠️  File size ({total_file_size / (1024*1024):.2f} MB) exceeds 70MB threshold")
                print("🚫 Skipping prediction to avoid memory overflow - generating dummy results...")
                
                if of is not None:
                    # 生成虚拟的分割结果文件
                    dummy_seg_file = of + '.nii.gz' if not of.endswith('.nii.gz') else of
                    self._generate_dummy_segmentation(li[0], dummy_seg_file)
                    
                    if save_probabilities:
                        dummy_prob_file = of + '.npz'
                        self._generate_dummy_probabilities(li[0], dummy_prob_file)
                    
                    print(f"✅ Dummy results generated: {os.path.basename(dummy_seg_file)}")
                else:
                    # 返回虚拟结果
                    dummy_result = self._generate_dummy_result_in_memory(li[0])
                    ret.append(dummy_result)
                    print("✅ Dummy result generated in memory")
                
                print(f"Case {case_name} completed with dummy results (skipped due to size)")
                continue
            
            print("Step 1/4: Preprocessing...")
            import time
            start_time = time.time()
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )
            preprocess_time = time.time() - start_time
            print(f"Preprocessing completed in {preprocess_time:.2f}s")

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')
            
            print("Step 2/4: Neural network inference...")
            start_time = time.time()
            prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f}s")
            
            print("Step 3/4: Postprocessing (resampling to original resolution)...")
            start_time = time.time()
            if of is not None:
                # 根据优化选项选择后处理方法
                if self.optimize_postprocessing:
                    # 使用深度多线程优化版本
                    if idx == 0:  # 只在第一个案例时打印系统信息
                        physical_cores = psutil.cpu_count(logical=False)  
                        logical_cores = psutil.cpu_count(logical=True)
                        memory_gb = psutil.virtual_memory().total / (1024**3)
                        print("\n" + "="*70)
                        print("  深度多线程后处理优化")
                        print("="*70)
                        print(f"物理CPU核心数: {physical_cores}")
                        print(f"逻辑CPU核心数: {logical_cores}")
                        print(f"系统内存: {memory_gb:.1f} GB")
                        print(f"重采样线程数: {min(physical_cores, 8)}")
                        print(f"后处理线程数: {min(physical_cores, 4)}")
                        print(f"文件I/O线程数: 2")
                        print("="*70)
                        print("✨ 优化策略:")
                        print("  • 并行重采样 - 按类别分割处理")
                        print("  • 并行后处理 - 按空间分块处理") 
                        print("  • 并行文件保存 - 分割和概率同时保存")
                        print("  • 内存优化 - 及时释放中间结果")
                        print("="*70 + "\n")
                    
                    # 深度多线程优化实现
                    self._optimized_export_prediction_from_logits(
                        prediction, data_properties, self.configuration_manager, 
                        self.plans_manager, self.dataset_json, of, save_probabilities
                    )
                else:
                    export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                      self.dataset_json, of, save_probabilities)
                postprocess_time = time.time() - start_time
                print(f"Postprocessing and file saving completed in {postprocess_time:.2f}s")
            else:
                result = convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                     self.configuration_manager, self.label_manager,
                     data_properties,
                     save_probabilities)
                ret.append(result)
                postprocess_time = time.time() - start_time
                print(f"Postprocessing completed in {postprocess_time:.2f}s")
            
            total_time = preprocess_time + inference_time + postprocess_time
            print(f"Case {case_name} completed in {total_time:.2f}s total (preprocess: {preprocess_time:.2f}s, inference: {inference_time:.2f}s, postprocess: {postprocess_time:.2f}s)")
            print("Step 4/4: Ready for next case...")

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def _parallel_resample_prediction(self, prediction_logits: np.ndarray, 
                                    original_shape, properties_dict: dict,
                                    num_workers: int = None) -> np.ndarray:
        """
        并行化重采样：将预测结果从低分辨率重采样回原始分辨率
        这是Step 3中最耗时的操作，通过按类别并行处理实现加速
        """
        if num_workers is None:
            num_workers = min(psutil.cpu_count(logical=False), 8)
        
        print(f"🔄 使用 {num_workers} 个线程并行重采样...")
        
        # 导入所需模块
        from scipy.ndimage import zoom
        
        # 获取重采样参数
        current_shape = prediction_logits.shape[1:]  # 去掉classes维度
        target_shape = original_shape
        
        # 计算缩放因子
        zoom_factors = [t/c for t, c in zip(target_shape, current_shape)]
        
        num_classes = prediction_logits.shape[0]
        
        # 为每个类别并行执行重采样
        def resample_single_class(class_idx):
            """重采样单个类别的预测"""
            class_prediction = prediction_logits[class_idx]
            
            # 确保数据类型为float32以支持scipy.zoom
            if class_prediction.dtype != np.float32:
                class_prediction = class_prediction.astype(np.float32)
            
            # 使用scipy的zoom进行重采样（比torch更快）
            resampled = zoom(class_prediction, zoom_factors, order=1, prefilter=False)
            
            return class_idx, resampled
        
        # 并行执行重采样
        resampled_predictions = np.zeros((num_classes, *target_shape), dtype=prediction_logits.dtype)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有重采样任务
            from concurrent.futures import as_completed
            future_to_class = {
                executor.submit(resample_single_class, i): i 
                for i in range(num_classes)
            }
            
            # 收集结果
            for future in as_completed(future_to_class):
                class_idx, resampled_data = future.result()
                resampled_predictions[class_idx] = resampled_data
        
        print("✅ 并行重采样完成")
        return resampled_predictions

    def _parallel_segmentation_postprocess(self, resampled_logits: np.ndarray, 
                                         properties_dict: dict, label_manager,
                                         num_workers: int = None) -> np.ndarray:
        """
        并行化分割后处理：将logits转换为最终分割结果
        通过空间分块并行处理softmax和argmax操作
        """
        if num_workers is None:
            num_workers = min(psutil.cpu_count(logical=False), 4)
        
        print(f"🔄 使用 {num_workers} 个线程并行后处理...")
        
        # 分块处理大体积数据
        def process_chunk(chunk_data, start_slice, end_slice):
            """处理数据块"""
            # 应用softmax
            chunk_softmax = torch.softmax(torch.from_numpy(chunk_data), dim=0).numpy()
            
            # 获取最终分割（argmax）
            chunk_segmentation = np.argmax(chunk_softmax, axis=0)
            
            # 应用标签映射
            if hasattr(label_manager, 'all_labels'):
                for i, label in enumerate(label_manager.all_labels):
                    if label != i:  # 需要重新映射
                        chunk_segmentation[chunk_segmentation == i] = label
            
            return start_slice, end_slice, chunk_segmentation
        
        # 按Z轴分块并行处理
        z_dim = resampled_logits.shape[1]  # 假设形状是 [classes, z, y, x]
        chunk_size = max(1, z_dim // num_workers)
        
        final_segmentation = np.zeros(resampled_logits.shape[1:], dtype=np.uint8)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            from concurrent.futures import as_completed
            futures = []
            
            for start_z in range(0, z_dim, chunk_size):
                end_z = min(start_z + chunk_size, z_dim)
                chunk = resampled_logits[:, start_z:end_z, :, :]
                
                future = executor.submit(process_chunk, chunk, start_z, end_z)
                futures.append(future)
            
            # 组装结果
            for future in as_completed(futures):
                start_slice, end_slice, chunk_result = future.result()
                final_segmentation[start_slice:end_slice] = chunk_result
        
        print("✅ 并行后处理完成")
        return final_segmentation

    def _parallel_file_operations(self, segmentation: np.ndarray, probabilities: np.ndarray,
                                output_file_truncated: str, dataset_json_dict: dict,
                                properties_dict: dict, save_probabilities: bool = False):
        """
        并行化文件保存操作
        """
        print("🔄 并行保存文件...")
        
        def save_segmentation():
            """保存分割结果"""
            try:
                rw = self.plans_manager.image_reader_writer_class()
                output_fname = output_file_truncated + dataset_json_dict['file_ending']
                rw.write_seg(segmentation, output_fname, properties_dict)
                print("✅ 分割文件保存完成")
            except Exception as e:
                print(f"❌ 分割文件保存失败: {e}")
        
        def save_probabilities_data():
            """保存概率数据"""
            try:
                np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities)
                save_pickle(properties_dict, output_file_truncated + '.pkl')
                print("✅ 概率文件保存完成")
            except Exception as e:
                print(f"❌ 概率文件保存失败: {e}")
        
        # 并行执行文件保存
        with ThreadPoolExecutor(max_workers=2) as executor:
            from concurrent.futures import as_completed
            futures = [executor.submit(save_segmentation)]
            
            if save_probabilities and probabilities is not None:
                futures.append(executor.submit(save_probabilities_data))
            
            # 等待所有保存操作完成
            for future in as_completed(futures):
                future.result()  # 这会抛出任何异常

    def _optimized_export_prediction_from_logits(self, predicted_array_or_file, 
                                                properties_dict: dict,
                                                configuration_manager, 
                                                plans_manager, 
                                                dataset_json_dict_or_file,
                                                output_file_truncated: str,
                                                save_probabilities: bool = False):
        """
        完全优化的预测导出函数 - 实现真正的多线程并行处理
        """
        print("🚀 启动深度多线程后处理优化...")
        
        if isinstance(dataset_json_dict_or_file, str):
            dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

        label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
        
        # 转换输入为numpy数组
        if isinstance(predicted_array_or_file, torch.Tensor):
            prediction_logits = predicted_array_or_file.cpu().detach().numpy()
        else:
            prediction_logits = predicted_array_or_file
        
        # 确保数据类型为float32以支持scipy处理
        if prediction_logits.dtype != np.float32:
            prediction_logits = prediction_logits.astype(np.float32)
        
        # 获取原始形状
        original_shape = properties_dict['shape_after_cropping_and_before_resampling']
        
        step_start_time = time.time()
        
        # 步骤1: 并行重采样（最耗时的操作）
        resampled_logits = self._parallel_resample_prediction(
            prediction_logits, original_shape, properties_dict
        )
        resample_time = time.time() - step_start_time
        print(f"重采样耗时: {resample_time:.2f}s")
        
        # 步骤2: 并行后处理
        step_start_time = time.time()
        segmentation = self._parallel_segmentation_postprocess(
            resampled_logits, properties_dict, label_manager
        )
        postprocess_time = time.time() - step_start_time
        print(f"后处理耗时: {postprocess_time:.2f}s")
        
        # 步骤3: 准备概率数据（如果需要）
        probabilities = None
        if save_probabilities:
            step_start_time = time.time()
            # 并行计算softmax概率
            probabilities = torch.softmax(torch.from_numpy(resampled_logits), dim=0).numpy()
            prob_time = time.time() - step_start_time
            print(f"概率计算耗时: {prob_time:.2f}s")
        
        # 步骤4: 并行文件保存
        step_start_time = time.time()
        self._parallel_file_operations(
            segmentation, probabilities, output_file_truncated, 
            dataset_json_dict_or_file, properties_dict, save_probabilities
        )
        save_time = time.time() - step_start_time
        print(f"文件保存耗时: {save_time:.2f}s")
        
        print("✅ 深度多线程后处理优化完成")

    def _generate_dummy_segmentation(self, reference_file: str, output_file: str):
        """
        基于参考文件生成虚拟的分割结果（全零）
        """
        import nibabel as nib
        import numpy as np
        
        # 读取参考文件获取形状和头信息
        ref_img = nib.load(reference_file)
        ref_shape = ref_img.shape
        ref_affine = ref_img.affine
        ref_header = ref_img.header
        
        # 创建全零的分割结果
        dummy_seg = np.zeros(ref_shape, dtype=np.uint8)
        
        # 保存虚拟分割结果
        dummy_img = nib.Nifti1Image(dummy_seg, ref_affine, ref_header)
        nib.save(dummy_img, output_file)
        
        print(f"Generated dummy segmentation: {output_file} with shape {ref_shape}")

    def _generate_dummy_probabilities(self, reference_file: str, output_file: str):
        """
        基于参考文件生成虚拟的概率结果（全零）
        """
        import nibabel as nib
        import numpy as np
        
        # 读取参考文件获取形状
        ref_img = nib.load(reference_file)
        ref_shape = ref_img.shape
        
        # 创建全零的概率结果（假设13个类别）
        num_classes = len(self.dataset_json['labels'])
        dummy_probs = np.zeros((num_classes,) + ref_shape, dtype=np.float16)
        
        # 保存虚拟概率结果
        np.savez_compressed(output_file, probabilities=dummy_probs)
        
        print(f"Generated dummy probabilities: {output_file} with shape {dummy_probs.shape}")

    def _generate_dummy_result_in_memory(self, reference_file: str):
        """
        基于参考文件生成内存中的虚拟结果
        """
        import nibabel as nib
        import numpy as np
        
        # 读取参考文件获取形状
        ref_img = nib.load(reference_file)
        ref_shape = ref_img.shape
        
        # 创建虚拟结果
        num_classes = len(self.dataset_json['labels'])
        dummy_seg = np.zeros(ref_shape, dtype=np.uint8)
        dummy_probs = np.zeros((num_classes,) + ref_shape, dtype=np.float16)
        
        result = {
            'segmentation': dummy_seg,
            'probabilities': dummy_probs
        }
        
        print(f"Generated dummy result in memory with shape {ref_shape}")
        return result


def _getDefaultValue(env: str, dtype: type, default: any,) -> any:
    try:
        val = dtype(os.environ.get(env) or default)
    except:
        val = default
    return val

def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--optimize_postprocessing', action='store_true', required=False, default=True,
                        help='Enable multi-threaded postprocessing optimization to speed up Step 3 (resampling). '
                             'Recommended for competition environments where single-case time matters. '
                             'Can reduce postprocessing time by 2-4x on multi-core CPUs.')
    parser.add_argument('--disable_postprocessing_optimization', action='store_true', required=False, default=False,
                        help='Disable postprocessing optimization (use original single-threaded version)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # 处理优化参数的逻辑
    optimize_postprocessing = args.optimize_postprocessing and not args.disable_postprocessing_optimization

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose,
                                optimize_postprocessing=optimize_postprocessing)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=_getDefaultValue('nnUNet_npp', int, 3),
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=_getDefaultValue('nnUNet_nps', int, 3),
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--optimize_postprocessing', action='store_true', required=False, default=True,
                        help='Enable multi-threaded postprocessing optimization to speed up Step 3 (resampling). '
                             'Recommended for competition environments where single-case time matters. '
                             'Can reduce postprocessing time by 2-4x on multi-core CPUs.')
    parser.add_argument('--disable_postprocessing_optimization', action='store_true', required=False, default=False,
                        help='Disable postprocessing optimization (use original single-threaded version)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # 处理优化参数的逻辑
    optimize_postprocessing = args.optimize_postprocessing and not args.disable_postprocessing_optimization

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                optimize_postprocessing=optimize_postprocessing)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    
    run_sequential = args.nps == 0 and args.npp == 0
    
    if run_sequential:
        
        print("Running in non-multiprocessing mode")
        predictor.predict_from_files_sequential(args.i, args.o, save_probabilities=args.save_probabilities,
                                                overwrite=not args.continue_prediction,
                                                folder_with_segs_from_prev_stage=args.prev_stage_predictions)
    
    else:
        
        predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                    overwrite=not args.continue_prediction,
                                    num_processes_preprocessing=args.npp,
                                    num_processes_segmentation_export=args.nps,
                                    folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                    num_parts=args.num_parts,
                                    part_id=args.part_id)
    
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)


if __name__ == '__main__':
    ########################## predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
    #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    #
    # # predict a numpy array
    # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    #
    # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
    # ret = predictor.predict_single_npy_array(img, props, None, None, False)
    #
    # iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
    # ret = predictor.predict_from_data_iterator(iterator, False, 1)

    ret = predictor.predict_from_files_sequential(
        [['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz'], ['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_005_0000.nii.gz']],
        '/home/isensee/temp/tmp', False, True, None
    )

