"""
zi2zi-JiT 训练新字体 —— 统一参数配置（训练不同字体时，只改这一个文件）

对应 HanziGen 的 "Cell 0"：想训练一个新的字体，绝大多数情况下只需修改本文件，
无需再逐个去翻 lora_single_gpu_finetune_jit.py / scripts/generate_font_dataset.py
/ generate_chars.py 等脚本的参数。

使用方法（二选一）：
  1. 命令行一键流程：  python run_pipeline.py
  2. CloudStudio：打开 zi2zi_jit_cloudstudio.ipynb，将第 0 个 Cell（参数区）
     与本文件保持一致的取值。

目录约定（参照腾讯云"可视化目录"标准）：
  - fonts/   （可视化）把你想要训练的目标字体文件（.ttf/.otf）上传到这里
  - models/  （可视化）把 README 里的预训练模型（zi2zi-JiT-B-16.pth）放到这里
  - data/    （可视化）自动生成的训练/测试数据集
  - outputs/ （可视化）训练产物 + 推理生成的 PNG，导出打包后也放在这里
  不要在系统目录（/root、/tmp 等不可视化目录）存放重要文件，实例回收会丢失。
"""

import os

# ===========================================================================
# 0. 运行开关（不需要的步骤可以关掉，例如已有数据集时只做训练+导出）
# ===========================================================================
DO_DATA_PREP = True      # 从字体文件生成数据集
DO_TRAIN = True          # LoRA 微调训练
DO_GENERATE = True       # 用训练好的 checkpoint 推理生成 PNG
DO_EXPORT = True         # 把生成的 PNG 打包成 zip（导出下载）

# ===========================================================================
# 1. 导入：路径与素材（字体 / 预训练模型都放到"可视化目录"里）
# ===========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(PROJECT_ROOT, "fonts")        # 目标字体上传目录（可视化）
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")      # 预训练模型目录（可视化）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")    # 最终 zip 保存位置（可视化）

# 源字体：用于生成 content 图像（统一书写风格的参照字体）。
# 如果留空，则自动使用 FONTS_DIR 里唯一的一个字体作为源字体。
SOURCE_FONT = ""   # 例如 "fonts/SomeSourceFont.ttf"（可留空自动选择）

# 要训练的目标字体：FONTS_DIR 下的文件名（可以只写一个，也可以写多个）。
# 多个字体 = 多字体风格训练，适合演示字体间转换；单字体是日常"学一个新字体"。
TARGET_FONTS = ["fonts/MyTargetFont.ttf"]

# 预训练基座模型（README 下载链接里的文件，放到 MODELS_DIR 下）。
# README 提供两个模型（B / L 两个变体），"二选一"使用，脚本不会自动切换：
#   - 下载 zi2zi-JiT-B-16.pth -> 这里写 "models/zi2zi-JiT-B-16.pth"，且 MODEL="JiT-B/16"、CFG=2.6
#   - 下载 zi2zi-JiT-L-16.pth -> 这里写 "models/zi2zi-JiT-L-16.pth"，且 MODEL="JiT-L/16"、CFG=2.4
# 支持三种写法：文件名 / 相对路径 / 目录（自动取目录下 checkpoint-last.pth）。
BASE_CHECKPOINT = "models/zi2zi-JiT-B-16.pth"

# ===========================================================================
# 2. 数据准备（scripts/generate_font_dataset.py 的参数）
# ===========================================================================
# 字符集：本项目支持 gb2312 / gbk / big5 / jisx0208 / ksx1001（见 data_processing/charsets.py）。
# "gbk" 是真 GBK：20,902 个汉字（CJK 基本区，与 HanziGen 的 gbk 基准一致，Python 内置编码器生成）。
# 用真 GBK 补字时请同步：CHARSET="gbk"、NUM_CHARS=30000、MAX_CHARS_PER_FONT=None；
# 注意训练样本量约为 gb2312 全集的 3 倍，训练时间显著变长。
CHARSET = "gb2312"
# 每个字体从 CHARSET 里抽多少个字符进入训练集。字符越多，风格学得越全；
# 想让"缺失字补集"效果最好，建议 >= 字符集大小（gb2312 用 6763；真 GBK 用 20902）并配 MAX_CHARS_PER_FONT=None。
TRAIN_CHARS_PER_FONT = 500
# 每个字体抽多少个"训练未见过"的字符进测试集（只用于评估泛化/出图展示，不参与训练，不影响风格）
TEST_CHARS_PER_FONT = 8
RESOLUTION = 256            # 字形渲染分辨率，须与训练 IMG_SIZE 一致
TRAIN_SEED = 42
TEST_SEED = 99999
NUM_WORKERS_DATA_PREP = 4   # 数据集生成时的并行进程数

# ===========================================================================
# 3. LoRA 训练（lora_single_gpu_finetune_jit.py 的参数）
# ===========================================================================
MODEL = "JiT-B/16"            # JiT-B/16 或 JiT-L/16，须与你下载的预训练模型对应
IMG_SIZE = 256                # 训练图像分辨率（模型锁死 256，勿改）
NUM_FONTS = 1000              # 字体嵌入维度：必须与预训练模型一致（1000），改小/改大都会加载失败
NUM_CHARS = 20000             # 字符嵌入上界：只需 >= 数据集字符数。默认 20000 够用；
                              # 只有用"真 GBK 全字符集"（约 2.1 万汉字）时才需要调大到 30000
MAX_CHARS_PER_FONT = 200      # 训练时每个字体实际使用的字符数上限（None = 全部）；补集建议 None

# LoRA 超参（决定"学到的新字体的容量"）
LORA_R = 32                   # rank，越大容量越大；显存影响很小（二阶小量）
LORA_ALPHA = 32               # 缩放系数
LORA_TARGETS = "qkv,proj,w12,w3"   # 注入 LoRA 的目标层
LORA_DROPOUT = 0.0            # LoRA dropout
PROJ_DROPOUT = 0.1            # 模型原有投影 dropout

# 训练步数与学习率
EPOCHS = 200                  # 总轮数（收敛后可提前结束）
BLR = 8e-4                    # base learning rate（README 示例值）
MIN_LR = 1e-6
WARMUP_EPOCHS = 1
SAVE_LAST_FREQ = 10           # 每多少轮保存一次 checkpoint-last.pth
SEED = 42

# 扩散噪声参数（以下是基于 EDM 通用经验的方向性建议，非官方说明）：
#   笔画乱、错字多 -> NOISE_SCALE 降到 0.8（让去噪任务更保守）
#   模糊、缺乏细节 -> NOISE_SCALE 提到 1.2（增强噪声扰动）
#   P_MEAN / P_STD 一般不调：作者已按字体任务把 EDM 的 -1.2/1.2 调到 -0.8/0.8
# 重要：NOISE_SCALE 是"训练期"参数，会固化进 checkpoint，生成阶段无法覆盖，调整须重新训练。
P_MEAN = -0.8
P_STD = 0.8
NOISE_SCALE = 1.0
CFG = 2.6                     # classifier-free guidance：JiT-B/16 用 2.6，JiT-L/16 用 2.4

# 在线评估（训练中定期生成样例图）
ONLINE_EVAL = True
EVAL_STEP_FOLDERS = True      # True = 每次评估单独存一个 step_{epoch} 子目录
EVAL_FREQ = 10                # 每多少轮评估一次
NUM_IMAGES = 6                # 每个字符生成几张（用于对比）

# 训练批量（如果 AUTO_TUNE=True 会被自动覆盖；这里仅作回退/上限参考）
BATCH_SIZE = 16
GEN_BSZ = 16

# ===========================================================================
# 4. 硬件自动调优（BATCH_SIZE / NUM_WORKERS / GEN_BSZ 自动推算）
# ===========================================================================
AUTO_TUNE = True              # True：自动推算 batch_size 等；False：用上面的固定值
TUNE_METHOD = "probe"         # "probe"=运行时实测（最准，会花几十秒）| "table"=静态标定表
TUNE_RESERVE = 0.85           # 显存安全系数（预留 15% 给缓存/碎片）
TUNE_MAX_BATCH = 128          # batch_size 上限
TUNE_MAX_GEN_BSZ = 32         # 推理批量上限
TUNE_NUM_WORKERS_CAP = 12     # DataLoader 最大并行数

# ===========================================================================
# 5. 推理生成（generate_chars.py 的参数）
# ===========================================================================
GENERATE_NUM_IMAGES = None    # 生成张数（None = test.npz 全部）
GENERATE_BATCH_SIZE = 64      # 推理批量（AUTO_TUNE 时会按显存自动收紧）
GENERATE_CFG = None           # None = 沿用 checkpoint 里的 cfg
GENERATE_SAMPLING_METHOD = None   # euler / heun / ab2，None = 沿用 checkpoint
GENERATE_NUM_SAMPLING_STEPS = None
GENERATE_PAIRWISE = None      # "src_gen" / "target_gen" / None

# ===========================================================================
# 6. 缺失字补集生成（scripts/generate_missing_chars.py 的参数）
#    用途：按 CHARSET 计算"目标字体缺失的字"，用训练好的模型补全这些字，
#    类似 HanziGen 的缺字补全。例如字体缺少某些简/繁体字时，可一键补齐。
# ===========================================================================
DO_MISSING_GEN = True            # 训练并生成完常规 PNG 后，是否继续补集
MISSING_CHARSET = None           # None = 沿用上面的 CHARSET
MISSING_BATCH_SIZE = 32          # 补集推理批量
MISSING_CFG = None               # None = 沿用 checkpoint 里的 cfg
MISSING_SAMPLING_METHOD = None   # euler / heun / ab2，None = 沿用 checkpoint
MISSING_NUM_SAMPLING_STEPS = None
MISSING_NUM_IMAGES = None        # None = 生成全部缺失字
MISSING_PAIRWISE = "src_gen"     # "src_gen"=输出 源字形|生成结果 对比图 / "none"
MISSING_REF_CHARS = ""           # 逗号分隔的样式参考字（留空自动从目标字体挑）

# ===========================================================================
# 7. 导出（PNG 打包下载）
# ===========================================================================
EXPORT_PREFIX = "zi2zi_jit"   # 生成的 zip 名前缀
EXPORT_INCLUDE_CHECKPOINT = True   # 是否把 LoRA checkpoint 一并打进 zip
EXPORT_ZIP = True             # 是否打包成 zip（False 则只保留目录）

# ===========================================================================
# 8. 训练输出目录名（一个字体一套输出，避免互相覆盖）
# ===========================================================================
def _target_stem():
    """取第一个目标字体的文件名（不含扩展名）作为输出目录标识。"""
    if not TARGET_FONTS:
        return "font"
    name = os.path.basename(TARGET_FONTS[0])
    return os.path.splitext(name)[0]

FONT_TAG = _target_stem()
DATASET_DIR = os.path.join(DATA_DIR, FONT_TAG)          # 生成的数据集目录
TRAIN_DIR = os.path.join(DATASET_DIR, "train")          # 训练图像目录
TEST_NPZ_PATH = os.path.join(DATASET_DIR, "test.npz")   # 测试 npz
OUTPUT_DIR = os.path.join(OUTPUTS_DIR, FONT_TAG)        # 训练 checkpoint 输出目录
GEN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "generated_chars")  # 推理 PNG 输出目录
MISSING_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "missing_chars")  # 缺失字补集输出目录


# ===========================================================================
# 9. 自检：在运行前给出明确提示（不阻断，只打印）
# ===========================================================================
def print_summary():
    print("=" * 60)
    print(" zi2zi-JiT 训练配置预览")
    print("=" * 60)
    print("  目标字体 :", TARGET_FONTS)
    print("  源字体   :", SOURCE_FONT or "(自动选择)")
    print("  基座模型 :", BASE_CHECKPOINT)
    print("  模型变体 :", MODEL, " @ ", IMG_SIZE, "px")
    print("  数据目录 :", DATASET_DIR)
    print("  输出目录 :", OUTPUT_DIR)
    print("  补集目录 :", MISSING_OUTPUT_DIR, "(DO_MISSING_GEN={})".format(DO_MISSING_GEN))
    print("  LoRA     : r={} alpha={} targets={}".format(LORA_R, LORA_ALPHA, LORA_TARGETS))
    print("  训练轮数 :", EPOCHS, " epochs, lr=", BLR)
    print("  自动调优 :", "on (method={})".format(TUNE_METHOD) if AUTO_TUNE else "off")
    print("=" * 60)
