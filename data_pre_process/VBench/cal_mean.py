import json

# 定义一个函数来计算平均值
def calculate_average(data):
    # 提取数值列表
    values = [item['video_results'] for item in data['aesthetic_quality'][1]]
    # 计算平均值
    average = sum(values) / len(values)
    return average

# 示例JSON数据
json_data = {'aesthetic_quality': (0.6737647950649261, [{'video_path': '/group/ossdphi_algo_scratch_01/hecui102/codelib/helps/VideoCrafter/results/base_512_v2/0001.mp4', 'video_results': 0.6279004216194153}, {'video_path': '/group/ossdphi_algo_scratch_01/hecui102/codelib/helps/VideoCrafter/results/base_512_v2/0002.mp4', 'video_results': 0.660439133644104}, {'video_path': '/group/ossdphi_algo_scratch_01/hecui102/codelib/helps/VideoCrafter/results/base_512_v2/0003.mp4', 'video_results': 0.6991678476333618}, {'video_path': '/group/ossdphi_algo_scratch_01/hecui102/codelib/helps/VideoCrafter/results/base_512_v2/0004.mp4', 'video_results': 0.6918447613716125}, {'video_path': '/group/ossdphi_algo_scratch_01/hecui102/codelib/helps/VideoCrafter/results/base_512_v2/0005.mp4', 'video_results': 0.6820043921470642}, {'video_path': '/group/ossdphi_algo_scratch_01/hecui102/codelib/helps/VideoCrafter/results/base_512_v2/0006.mp4', 'video_results': 0.681232213973999}])}

# 计算并打印平均值
average_value = calculate_average(json_data)
print(f"The average value is: {average_value}")
