# accept rate
import re
def calculate_average_score(log_file_path):
    scores = []
    
    # 正则表达式用于匹配 Updated score 行
    score_pattern = re.compile(r'Updated score: (\d+)')
    
    with open(log_file_path, 'r') as file:
        for line in file:
            match = score_pattern.search(line)
            if match:
                score = int(match.group(1))
                scores.append(score)

    if not scores:
        print("没有找到任何 'Updated score' 条目")
        return 0
    
    average_score = sum(scores) / len(scores)
    # print(len(scores))
    return average_score

# 示例使用
log_file_path = 'log/openr_base_env_log_2025-01-30_23-54-11.log'  # 将此路径替换为您的日志文件的实际路径
average_score = calculate_average_score(log_file_path)
print(f"日志中 'Updated score' 的平均值是: {average_score/6}")

total_time_sum = 0.0
generation_times = []
evaluation_times = []

with open(log_file_path, 'r') as file:
    for line in file:
        # 检查是否包含Updated problem_time
        if 'Updated problem_time:' in line:
            # 找到冒号的位置
            colon_pos = line.find('Updated problem_time:')
            # 提取时间值并累加到total_time_sum
            total_time_sum += float(line[colon_pos + len('Updated problem_time: '):])
        
        # 检查是否包含Updated time
        elif 'Updated time:' in line:
            # 找到冒号的位置
            colon_pos = line.find('Updated time:')
            # 提取列表形式的时间值
            times_str = line[colon_pos + len('Updated time: ['):]
            # 移除末尾的']'和可能存在的换行符
            times_str = times_str.rstrip().rstrip(']')
            # 将字符串分割成generation time和evaluation time
            gen_time, eval_time = map(float, times_str.split(', '))
            generation_times.append(gen_time)
            evaluation_times.append(eval_time)

# 计算总时间（如果需要的话，可以是total_time加上generation和evaluation时间）
overall_total_time = total_time_sum + sum(generation_times) + sum(evaluation_times)

# 计算平均值
average_total_time = total_time_sum / max(len(generation_times), 1)  # 防止除以0
average_generation_time = sum(generation_times) / max(len(generation_times), 1)
average_evaluation_time = sum(evaluation_times) / max(len(evaluation_times), 1)

print(f"Total Time calculated from 'Updated problem_time': {total_time_sum}")
print(f"Sum of Generation Time from 'Updated time': {sum(generation_times)}")
print(f"Sum of Evaluation Time from 'Updated time': {sum(evaluation_times)}")
print(f"Overall Total Time (Sum of all times): {overall_total_time}")

print("\nAverages:")
print(f"Average Total Time per record: {average_total_time}")
print(f"Average Generation Time: {average_generation_time}")
print(f"Average Evaluation Time: {average_evaluation_time}")

