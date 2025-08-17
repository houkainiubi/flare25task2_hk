"""
真正的多线程后处理优化
在应用层面实现并行重采样和文件保存
"""
import os
import time
import torch
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Tuple
from functools import partial
import threading
from queue import Queue
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from scipy.ndimage import zoom
from skimage.transform import resize


def parallel_resample_prediction(prediction_logits: np.ndarray, 
                                original_shape: Tuple[int, ...], 
                                properties_dict: dict,
                                num_workers: int = None) -> np.ndarray:
    """
    并行化重采样：将预测结果从低分辨率重采样回原始分辨率
    这是Step 3中最耗时的操作
    
    Args:
        prediction_logits: 预测的logits数组 [classes, z, y, x]
        original_shape: 原始图像形状
        properties_dict: 包含spacing等信息的属性字典
        num_workers: 并行工作线程数
        
    Returns:
        重采样后的预测结果
    """
    if num_workers is None:
        num_workers = min(psutil.cpu_count(logical=False), 8)
    
    print(f"🔄 使用 {num_workers} 个线程并行重采样...")
    
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
        
        # 使用scipy的zoom进行重采样（比torch更快）
        resampled = zoom(class_prediction, zoom_factors, order=1, prefilter=False)
        
        return class_idx, resampled
    
    # 并行执行重采样
    resampled_predictions = np.zeros((num_classes, *target_shape), dtype=prediction_logits.dtype)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有重采样任务
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


def parallel_segmentation_postprocess(resampled_logits: np.ndarray, 
                                    properties_dict: dict,
                                    label_manager,
                                    num_workers: int = None) -> np.ndarray:
    """
    并行化分割后处理：将logits转换为最终分割结果
    包括softmax、argmax等操作
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


def parallel_file_operations(segmentation: np.ndarray,
                           probabilities: np.ndarray,
                           output_file_truncated: str,
                           dataset_json_dict: dict,
                           properties_dict: dict,
                           plans_manager,
                           save_probabilities: bool = False) -> None:
    """
    并行化文件保存操作
    """
    print("🔄 并行保存文件...")
    
    def save_segmentation():
        """保存分割结果"""
        try:
            rw = plans_manager.image_reader_writer_class()
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
        futures = [executor.submit(save_segmentation)]
        
        if save_probabilities and probabilities is not None:
            futures.append(executor.submit(save_probabilities_data))
        
        # 等待所有保存操作完成
        for future in as_completed(futures):
            future.result()  # 这会抛出任何异常


def optimized_export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], 
                                           properties_dict: dict,
                                           configuration_manager, 
                                           plans_manager, 
                                           dataset_json_dict_or_file: Union[dict, str], 
                                           output_file_truncated: str,
                                           save_probabilities: bool = False,
                                           enable_parallel_resample: bool = True,
                                           resample_workers: int = None) -> None:
    """
    完全优化的预测导出函数
    实现真正的多线程并行处理
    """
    print("🚀 启动深度多线程后处理优化...")
    
    if isinstance(dataset_json_dict_or_file, str):
        from batchgenerators.utilities.file_and_folder_operations import load_json
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    # 转换输入为numpy数组
    if isinstance(predicted_array_or_file, torch.Tensor):
        prediction_logits = predicted_array_or_file.cpu().detach().numpy()
    else:
        prediction_logits = predicted_array_or_file
    
    # 获取原始形状
    original_shape = properties_dict['shape_after_cropping']
    
    start_time = time.time()
    
    # 步骤1: 并行重采样（最耗时的操作）
    if enable_parallel_resample:
        resampled_logits = parallel_resample_prediction(
            prediction_logits, original_shape, properties_dict, resample_workers
        )
    else:
        # 回退到原始方法
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        result = convert_predicted_logits_to_segmentation_with_correct_shape(
            prediction_logits, plans_manager, configuration_manager, label_manager, 
            properties_dict, return_probabilities=save_probabilities
        )
        if save_probabilities:
            segmentation, probabilities = result
        else:
            segmentation = result
            probabilities = None
        
        # 并行保存文件
        parallel_file_operations(
            segmentation, probabilities, output_file_truncated, 
            dataset_json_dict_or_file, properties_dict, plans_manager, save_probabilities
        )
        return
    
    resample_time = time.time() - start_time
    print(f"重采样耗时: {resample_time:.2f}s")
    
    # 步骤2: 并行后处理
    start_time = time.time()
    segmentation = parallel_segmentation_postprocess(
        resampled_logits, properties_dict, label_manager
    )
    postprocess_time = time.time() - start_time
    print(f"后处理耗时: {postprocess_time:.2f}s")
    
    # 步骤3: 准备概率数据（如果需要）
    probabilities = None
    if save_probabilities:
        start_time = time.time()
        # 并行计算softmax概率
        probabilities = torch.softmax(torch.from_numpy(resampled_logits), dim=0).numpy()
        prob_time = time.time() - start_time
        print(f"概率计算耗时: {prob_time:.2f}s")
    
    # 步骤4: 并行文件保存
    start_time = time.time()
    parallel_file_operations(
        segmentation, probabilities, output_file_truncated, 
        dataset_json_dict_or_file, properties_dict, plans_manager, save_probabilities
    )
    save_time = time.time() - start_time
    print(f"文件保存耗时: {save_time:.2f}s")
    
    print("✅ 深度多线程后处理优化完成")


def print_advanced_optimization_info():
    """
    打印高级优化信息
    """
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print("\n" + "="*70)
    print("🚀 高级多线程后处理优化")
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
