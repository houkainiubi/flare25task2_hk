"""
后处理优化模块
专门针对比赛环境优化Step 3的重采样和文件保存速度
"""
import os
import torch
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Union
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle


def export_prediction_from_logits_threaded(predicted_array_or_file: Union[np.ndarray, torch.Tensor], 
                                         properties_dict: dict,
                                         configuration_manager, 
                                         plans_manager, 
                                         dataset_json_dict_or_file: Union[dict, str], 
                                         output_file_truncated: str,
                                         save_probabilities: bool = False,
                                         num_threads_torch: int = None,
                                         enable_cc3d_optimization: bool = True):
    """
    多线程优化版本的export_prediction_from_logits
    充分利用CPU资源进行后处理加速
    
    优化策略：
    1. 使用更多torch线程进行重采样（CPU密集型操作）
    2. 并行执行文件保存任务
    3. 智能线程数配置
    4. cc3d连通组件优化
    """
    if num_threads_torch is None:
        # 获取物理CPU核心数，但限制最大线程数避免内存压力
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        # 对于重采样这种CPU密集型任务，使用物理核心数+50%的超线程
        num_threads_torch = min(int(physical_cores * 1.5), 16)
    
    print(f"🚀 使用 {num_threads_torch} 个线程进行后处理优化...")
    
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    # 设置torch线程数进行优化
    old_threads = torch.get_num_threads()
    # 重采样是CPU密集型，可以用更多线程
    torch.set_num_threads(num_threads_torch)
    
    try:
        # 导入转换函数
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
            return_probabilities=save_probabilities
        )
        del predicted_array_or_file

        # cc3d优化（如果启用）
        if enable_cc3d_optimization:
            try:
                if save_probabilities:
                    segmentation_final, probabilities_final = ret
                    segmentation_final = optimize_segmentation_with_cc3d(segmentation_final)
                    ret = (segmentation_final, probabilities_final)
                    print("✅ cc3d连通组件优化完成")
                else:
                    segmentation_final = ret
                    segmentation_final = optimize_segmentation_with_cc3d(segmentation_final)
                    ret = segmentation_final
                    print("✅ cc3d连通组件优化完成")
            except ImportError:
                print("⚠️  cc3d库未安装，跳过连通组件优化")
            except Exception as e:
                print(f"⚠️  cc3d优化失败，使用原始结果: {e}")

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


def optimize_segmentation_with_cc3d(segmentation: np.ndarray):
    """
    使用cc3d库优化分割结果的连通组件
    """
    try:
        import cc3d
        
        # 获取独特的标签（排除背景0）
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels != 0]
        
        optimized_segmentation = segmentation.copy()
        
        for label in unique_labels:
            # 提取当前标签的mask
            mask = (segmentation == label).astype(np.uint8)
            
            # 使用cc3d进行连通组件分析
            labels_out = cc3d.connected_components(mask, connectivity=26)  # 3D 26-connectivity
            
            # 保留最大的连通组件
            if labels_out.max() > 1:  # 如果有多个连通组件
                # 计算每个组件的体积
                component_sizes = np.bincount(labels_out.flat)
                component_sizes[0] = 0  # 忽略背景
                
                # 找到最大的组件
                largest_component = np.argmax(component_sizes)
                
                # 只保留最大组件
                largest_mask = (labels_out == largest_component)
                optimized_segmentation[mask.astype(bool)] = 0  # 清除原始标签
                optimized_segmentation[largest_mask] = label  # 设置优化后的标签
        
        return optimized_segmentation
        
    except ImportError:
        # 如果cc3d不可用，返回原始分割
        return segmentation
    except Exception as e:
        print(f"cc3d优化过程中出现错误: {e}")
        return segmentation


def get_optimal_thread_count():
    """
    根据系统配置获取最优线程数
    """
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 基于CPU和内存配置的启发式规则
    if memory_gb >= 32:  # 大内存系统
        optimal_threads = min(logical_cores, 20)
    elif memory_gb >= 16:  # 中等内存系统  
        optimal_threads = min(int(physical_cores * 1.5), 16)
    else:  # 小内存系统
        optimal_threads = min(physical_cores, 8)
    
    return optimal_threads


def print_optimization_info():
    """
    打印系统信息和优化建议
    """
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    optimal_threads = get_optimal_thread_count()
    
    print("\n" + "="*60)
    print("🖥️  系统配置信息")
    print("="*60)
    print(f"物理CPU核心数: {physical_cores}")
    print(f"逻辑CPU核心数: {logical_cores}")
    print(f"系统内存: {memory_gb:.1f} GB")
    print(f"推荐线程数: {optimal_threads}")
    
    # 检查cc3d可用性
    try:
        import cc3d
        print("cc3d库状态: ✅ 已安装 (连通组件优化可用)")
    except ImportError:
        print("cc3d库状态: ❌ 未安装 (建议安装: pip install connected-components-3d)")
    
    print("="*60)
    print("🚀 后处理优化已启用，将加速Step 3重采样过程")
    print("="*60 + "\n")
