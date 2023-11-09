import argparse
from pathlib import Path 
import time
import dcgm
import csv
import timeit


PROFILING_IDS = {
    "Graphics Engine Activity": 1001, 
    "SM Activity": 1002,
    "SM Occupancy": 1003,
    "Tensor Core Activity": 1004,
    "Memory BW Activity": 1005,
    "FP64 Engine Activity": 1006,
    "FP32 Engine Activity": 1007,
    "FP16 Engine Activity": 1008,
    "PCIe Bandwidth Tx": 1009,
    "PCIe Bandwidth Rx": 1010,
    "NVLink Bandwidth Tx": 1011,
    "NVLink Bandwidth Rx": 1012,
}

DEFAULT_PROFILING_IDS = [
    1003,  # SM Occupancy
    1004,  # Tensor Core Activity
    1005,  # Memory BW Activity
    1009,  # PCIe Bandwidth Tx
    1010,  # PCIe Bandwidth Rx
    1011,  # NVLink Bandwidth Tx
    1012,  # NVLink Bandwidth Rx
]

# starts daemon that monitors dcgm metrics and outputs to arrow file
if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-ids", nargs="+", type=int, default=DEFAULT_PROFILING_IDS,
                        help="dcgm field_ids to monitor")
    parser.add_argument("--sampling-interval", type=int, default=1000,
                        help="dcgm sampling interval in ms")
    parser.add_argument("--output-file", type=str, default="dcgm_metrics.csv",
                        help="output file for dcgm metrics")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite output file if it exists")
    parser.add_argument("--write-interval", type=int, default=3600,
                        help="how often to write to output file in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="print debug info to stdout")
    parser.add_argument("--quiet", action="store_true",
                        help="don't print anything to stdout")
    args = parser.parse_args()

    # 重定向print函数以根据参数决定是否输出
    # monkey patch print to be quiet
    _builtin_print = print

    def print(*pargs, **kwargs):
        if not args.quiet:
            _builtin_print(*pargs, **kwargs)

    # add verbose print
    # 添加带有verbose参数的print函数
    def printv(*pargs, **kwargs):
        if args.verbose: 
            print(*pargs, **kwargs)


    # 从命令行参数中获取字段ID、采样间隔和写入间隔等信息
    fieldIds = args.field_ids
    dcgmSamplingInterval = args.sampling_interval
    dcgmMaxKeepAge = args.write_interval

    # 初始化dcgm
    dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle = dcgm.init_hostengine(
        fieldIds=fieldIds,
        dcgmSamplingInterval=dcgmSamplingInterval,
        dcgmMaxKeepAge=dcgmMaxKeepAge,
    )

    # 检查输出文件是否存在，并根据参数决定是否覆盖
    outputFile = Path(args.output_file)
    if outputFile.exists() and not args.overwrite:
        raise ValueError(f"output file {outputFile} already exists")

    # 创建CSV写入器并写入表头
    writer = csv.writer(outputFile.open("w"))
    writer.writerow(["GPU Id", "Sample Step", "Field ID", "Value"])

    # 用于跟踪每个GPU和字段的采样步骤计数,0为初始值
    sample_step_counts = {
        gpu_id: {field_id: 0 for field_id in fieldIds}
        for gpu_id in dcgmGroup.GetGpuIds()
    }

    # 定义日志函数，用于记录dcgm指标到文件中
    def log(f, dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle):
        # 获取所有指标 # get all metrics
        dfvc = dcgm.get_metrics(dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle) 
        metrics = dfvc.values 

        # 遍历每个GPU # for each gpu
        for gpuId, gpuMetrics in metrics.items():
            # 遍历每个字段ID # for each field id
            for id in fieldIds:
                # 遍历每个采样值
                for sample in gpuMetrics[id].values:
                    writer.writerow([gpuId, sample_step_counts[gpuId][id], id, sample.value])
                    sample_step_counts[gpuId][id] += 1
                printv(f"\tLogged {len(gpuMetrics[id].values)} samples for {id}")

        dfvc.EmptyValues()
        return dfvc


    # 记录上一次写入时间
    last_write_time = timeit.default_timer()

    # 循环记录dcgm指标
    while True:
        print(
            f"Logging at {timeit.default_timer()} (total time since last: {timeit.default_timer() - last_write_time})")
        last_write_time = timeit.default_timer()
        dfvc = log(f, dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle)
        time.sleep(args.write_interval)








