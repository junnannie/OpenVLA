import os
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import torch
import numpy as np
import tqdm
from libero.libero import benchmark
import copy

import utils
from config import Config


config = Config()
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def eval_libero():

    assert config.pretrained_checkpoint is not None, "config.pretrained_checkpoint must not be None!"
    assert not (config.load_in_8bit and config.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # 加载模型
    model = AutoModelForVision2Seq.from_pretrained(
        config.pretrained_checkpoint,

        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,

        # 使用8-bit或4-bit量化
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,

        # 减少内存使用
        low_cpu_mem_usage=True,

        # 允许下载使用远程代码
        trust_remote_code=True,
    )

    # 加载 lora 权重
    if config.lora_checkpoint:
        model = PeftModel.from_pretrained(model, config.lora_checkpoint)
        print(f"Loaded lora weights from {config.lora_checkpoint}")

    # 不使用量化时，手动将模型加载到GPU上
    if not config.load_in_8bit and not config.load_in_4bit:
        model = model.to(DEVICE)
    
    # 1.norm_stats 怎么来的
    # 设置配置的 unnorm_key 等于模型的 norm_stats ，用于数据的归一化
    # task_suite_name: str = "libero_spatial"
    config.unnorm_key = config.task_suite_name

    assert config.unnorm_key in model.norm_stats, f"Action un-norm key {config.unnorm_key} not found in VLA `norm_stats`!"
    
    # 加载 processor , checkpoint =  "/mnt/workspace/openvla-7b-finetuned-libero-spatial"
    processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
    

    # 初始化日志
    run_id = f"EVAL-{config.task_suite_name}-{config.model_family}-{utils.DATE_TIME}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(config.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # 初始化LIBERO任务
    benchmark_dict = benchmark.get_benchmark_dict()

    # 获取指定的任务场景
    task_suite = benchmark_dict[config.task_suite_name]()

    # 指定场景中的任务数量
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Task suite: {config.task_suite_name}, task_num: {num_tasks_in_suite}")
    log_file.write(f"Task suite: {config.task_suite_name}, task_num: {num_tasks_in_suite}\n")

    # test
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        task = task_suite.get_task(task_id)

        # 从指定的场景中，获取第 i 个任务的初始状态
        initial_states = task_suite.get_task_init_states(task_id)
        # 任务每次执行时，物品位置略有差异，因此 initial_states 是个数组，表示每次初试状态的轻微差异

        # 初始化LIBERO仿真环境，并获得任务描述
        env, task_description = utils.get_libero_env(task, config.model_family, resolution=768)

        # 将上述任务执行 num_trials_per_task 次
        task_episodes, task_successes = 0, 0

        # num_trials_per_task: int = 2
        for episode_idx in tqdm.tqdm(range(config.num_trials_per_task)):

            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()

            # 设置初始状态
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            while t < config.max_steps:
                try:
                    # 前 10 个时间步不进行操作，因为物体可能正在掉落
                    if t < 10:

                        # step 传入一个 action，action 是一个长度为 7 的数组，分别表示机械臂的七个关节的角度
                        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    # 获取环境的观测
                    img = obs["agentview_image"]

                    # img.shape = (768, 768, 3)
                    # 将图片旋转180度，LIBERO仿真环境的相机是倒着安装的
                    img = img[::-1, ::-1]

                    # 将图片存入数组，用来生成视频
                    replay_images.append(copy.deepcopy(img))

                    # 缩放到模型视觉编码器的输入尺寸
                    img = utils.resize_image(img, (224, 224))
                    
                    # 构造提示词
                    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

                    # 通过处理器得到模型输入
                    inputs = processor(prompt, Image.fromarray(img).convert("RGB")).to(DEVICE, dtype=torch.bfloat16)

                    # 模型推理，得到下一步的动作
                    action = model.predict_action(**inputs, unnorm_key=config.unnorm_key, do_sample=False)
                    # action.shape = (1, 7)

                    # 处理输出的 action，使其符合 LIBERO 的定义（LIBERO 中夹爪的控制量：-1 = open, +1 = close）
                    # 将夹爪的控制量，从 [0, 1] 范围转换到 [-1, +1]
                    action[..., -1] = 2 * action[..., -1] - 1

                    # 将夹爪控制量进行二值化, 大于0的值置为 +1，小于0的值置为 -1
                    action[..., -1] = np.sign(action[..., -1])

                    # 将夹爪控制量反转
                    action[..., -1] = action[..., -1] * -1.0

                    # 更新环境
                    obs, reward, done, info = env.step(action.tolist())

                    # 是否完成
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存视频
            utils.save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # 添加日志
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # 添加总结日志
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    eval_libero()
