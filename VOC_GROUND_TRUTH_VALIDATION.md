# VOC Ground Truth Validation（VOC2007 官方回填核验）

> 日期：2026-09-05 ｜ 模式：最小官方下载 + 身份核验 + 官方 annotation 解析
> 范围：`data/db/smartimager.db` 66 行（60 唯一内容） ｜ 未改生产代码/图片/排序，未跑 GLM/R1/R2
> 结论先行：**60/60 唯一内容与官方 PASCAL VOC2007 文件字节级一致，官方 XML annotation 已获取，
> 正式 Retrieval Gold Set 可按 Strong Ground Truth 构建。**

报告中每条结论标注来源类型：**Fact**（代码/字节/官方文件实证）、**Inference**（推测）、
**Validation Result**（本次核验结果）。

---

## 1. 数据来源确认
| 结论 | 类型 | 说明 |
|---|---|---|
| 代码/文档中没有任何 VOC 来源记录、下载脚本或标注资产 | Fact | 全库检索 VOC2007/PASCAL/devkit 0 命中（模型 tokenizer 词表除外） |
| DB 内 66 行、60 个唯一内容（6 组字节重复，两组上传批次） | Fact | 本地 SHA-256 统计 |
| 文件名 `2007_XXXXXX.jpg` 与 PASCAL VOC 系列命名规则一致 | Inference→已升级为 Validation | 仅文件名是推断；经下方核验已被证实 |
| 图片可能被重编码（无 EXIF、~500px）→ 不可靠 | Inference | **被证伪**：官方 VOC2007 JPEG 同样无 EXIF，本地文件与官方 SHA-256 完全一致 |

## 2. 官方数据来源与最小下载方法
- 官方源：Oxford PASCAL VOC2007（`https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/`，
  最终下载主机 `thor.robots.ox.ac.uk`，HTTP Range 支持）。
- 文件：`VOCtrainval_06-Nov-2007.tar`（460,032,000 B）、`VOCtest_06-Nov-2007.tar`。
- 最小化手段：**不下载整库**。只做 16MB 窗口 Range 稀疏扫描（每库仅读约 64MB 即定位全部目标），
  最终仅保留 60 张官方 JPEG + 60 个官方 XML + 4 个 ImageSets/Main 全局清单（共约 6 MB）。
- 官方页面事实：VOC2007 共 9,963 张 / 24,640 个标注对象 / 20 类；每图 XML；2007 是最后一个公开 test 标注年份（2008+ test 不再公开）。

## 3. ID Mapping
DB `images.filename` = `2007_XXXXXX.jpg` → 本报告使用的 key：
- **annotation / JPEG 名**：6 位 id `XXXXXX`（官方 tar 内 `Annotations/XXXXXX.xml`、`JPEGImages/XXXXXX.jpg`，
  注意：官方 VOC2007 JPEG 文件名**不带** `2007_` 前缀，是纯 6 位数字）。
- 校验：66 行 → 60 个唯一 id，全部命中官方集合，无 both/neither。

## 4. Split 核验（Validation Result）
依据官方 `ImageSets/Main/{train,val,trainval,test}.txt`（从对应 tar 下载）：

| Split | 唯一内容数 | 说明 |
|---|---|---|
| train | 15 | trainval tar |
| val | 23 | trainval tar |
| test | 22 | test tar（VOC2007 test annotation 公开，可用） |
| 合计 | 60 | 无重叠 |

## 5. 图片身份核验方法与结果（Validation Result）
方法（双重）：
1. **字节级**：本地与官方 JPEG 的 SHA-256 对比。
2. **感知级**（满足“不得只用字节 hash”要求）：PIL 解码后比较
   - 尺寸/宽高比；
   - 96×96 灰度 resize 像素余弦相似度；
   - 平均绝对像素差；
   - 平均感知 hash Hamming 距离。

结果（60 个唯一内容，每个均取全部 DB 行与官方比对）：

| 指标 | 结果 |
|---|---|
| MATCH / LIKELY_MATCH / NO_MATCH / UNKNOWN | **60 / 0 / 0 / 0** |
| 余弦相似度 | 全部 1.0 |
| 平均绝对像素差 | 全部 0.0 |
| pHash Hamming | 全部 0 |
| 本地尺寸 == 官方尺寸 | 60/60（例：000664→500×375，000272→333×500，000733→450×435） |
| SHA-256 字节一致 | **60/60**（含 6 组重复行的每行） |

即：DB 中图片就是官方 VOC2007 JPEG 原件（未重编码/未缩放）。早期“无 EXIF/JFIF”观察与官方文件自身属性一致。

## 6. Annotation 获取（Validation Result）
- 官方 XML 60/60 已获取并解析（来自各自 split 对应 tar）。
- 每图提取**全部** `<object><name>`，保留多标签，例如：
  - `001027` → {bicycle, chair, sofa, tvmonitor}
  - `000346` → {bottle, person}
  - `000904` → {cow, horse, person}
- 60 张中 25 张单标签（1 个 class）、其余 2~4 个标签；平均每图 1.75 个 class。

## 7. Duplicate 处理
| 内容组 (6 位 id) | DB 重复行 |
|---|---|
| 000664 | 2, 37 |
| 000733 | 3, 42 |
| 000876 | 1, 57 |
| 000904 | 4, 58 |
| 001027 | 5, 61 |
| 001073 | 6, 62 |

建议 **Gold Set 按 unique image content 构建**（60 个 id），并在需要时映射回 DB rows（66）。
禁止把同一内容的两行重复计权。上述重复行字节与官方一致，映射无歧义。

## 8. Category Distribution（Validation Result，60 unique 口径）
20 个 VOC 类**全部出现**（该类均匀抽样非常罕见），官方 XML 逐图统计：

| class | n(unique) | 健康(2–15)? | class | n(unique) | 健康? |
|---|---|---|---|---|---|
| person | 26 | ✗（剔除） | cat | 4 | ✓ |
| sofa | 7 | ✓ | boat | 4 | ✓ |
| aeroplane | 6 | ✓ | tvmonitor | 4 | ✓ |
| horse | 6 | ✓ | diningtable | 4 | ✓ |
| bottle | 5 | ✓ | train | 3 | ✓ |
| chair | 5 | ✓ | motorbike | 3 | ✓ |
| cow | 5 | ✓ | pottedplant | 3 | ✓ |
| car | 5 | ✓ | bird | 2 | ✓ |
| bicycle | 5 | ✓ | sheep | 2 | ✓ |
| bus | 5 | ✓ | dog | 1 | ✗（n=1） |

覆盖 20 类；n≥2 有 19 类，n≥3 有 17 类，n≥5 有 10 类。

## 9. 自动生成的 Retrieval Gold Query（Validation Result）
仅由官方 annotation 生成（无 LLM/CLIP/VLM/聚类参与）；固定中文名词映射，语义直接对应 GT。

### 核心：18 条健康单类 Query（2≤relevant≤15）
| Query | relevant unique（→ 对应 DB rows） |
|---|---|
| 飞机的照片 | 6 |
| 马的照片 | 6 |
| 自行车的照片 | 5（rows=6，含重复） |
| 汽车的照片 | 5 |
| 公交车的照片 | 5 |
| 牛的照片 | 5（rows=7） |
| 瓶子的照片 | 5 |
| 椅子的照片 | 5（rows=6） |
| 沙发的照片 | 7（rows=8） |
| 猫的照片 | 4（rows=5） |
| 船的照片 | 4 |
| 电视或显示器的照片 | 4（rows=5） |
| 餐桌的照片 | 4 |
| 火车的照片 | 3 |
| 摩托车的照片 | 3（rows=4） |
| 盆栽的照片 | 3 |
| 鸟的照片 | 2 |
| 羊的照片 | 2 |

### 排除项
- `dog`（n=1）：仅统计，不构成核心 ranking Query。
- `person`（n=26，>60 的 40%）：接近退化，剔除。

### 可选组合
- 语义明确可用的「并集」Query（如 A∪B，相关=含任一 class）在 2~15 区间有 **171 个候选**；
  若需把 18 提升到 20–30 条，可从中选取极少量措辞明确（如“猫和狗的照片”）的组合，
  但**默认不强制填充**——避免为达标而引入弱语义。

## 10. Gold Set 可行性（Validation Result）
- MATCH 60；NO_MATCH 0；UNKNOWN 0；可用官方 annotation 60/60。
- 18 条健康单类 Query 已达标（2–15 relevant），离 20–30 目标差 2–12 条；
  加 2–12 条语义明确的组合 Query 即可满足 20–30（可选，见 §9）。
- **已达到 Strong Ground Truth 标准**：官方人工 annotation + 字节/感知双核验 MATCH。

## 11. 是否达到 Strong GT 标准
**是。** 仅以官方 VOC2007 annotation 为标签；图片身份核验全部 MATCH（双重校验），
标注来源独立于 SmartImager 的任何模型/管线。
Weak（filename 推断）/Pseudo（CLIP/VLM/cluster）在本流程中**未使用**。

## 12. 下一步建议
1. 以 60 个 unique content 生成正式 Gold Set JSON（`query → relevant content ids → db row ids`），
   写入 Evaluation/Benchmark 侧（仅改评估文件）。
2. 组合 Query 若启用：固定白名单（只允许明确并/交语义，禁止自由文本）。
3. 之后（需用户同意成本）才允许在**同一候选池**下运行 Baseline vs R1 vs R2 的 Rank Delta 评测。
4. 如需扩大 Query 规模：从官方 VOC2007 train/val 增补类别均衡图片（走现有上传流程）再重跑本流程。

---

## 附：临时验证物
本流程的下载与中间 JSON 暂存于系统 TEMP（`voc_dl/`、`voc_scan_*.json`、`voc_verify.json` 等，未进入仓库）；
报告数据可完全从 `VOC_GROUND_TRUTH_VALIDATION.md` 复述，仓库内除本文件外无新增/修改。
