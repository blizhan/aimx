# Feature Specification: Inline Terminal Image Rendering for `aimx query images`

**Feature Branch**: `003-query-images-terminal-render`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description: "`aimx query images` 命令通过 textual_image 支持直接显示图片 — rich.Console + textual_image.renderable.Image"

## Clarifications

### Session 2026-04-22

- Q: 如何**完全禁用**内联图片渲染？是否新增专门开关？ → A: 不新增开关；完全禁用图片复用既有机制（`--plain` / `--json` / 重定向非 TTY），`--max-images` 仅用于调上限。
- Q: 默认 `--max-images` 上限应设为多少？ → A: 6 张（默认值）。
- Q: 图片渲染与元数据行的版面关系？ → A: 保留顶部 rich 汇总表格，在表格之后按每条匹配分段打印"元数据头 + 下方图片"块。
- Q: 依赖缺失时 warning 的输出通道？ → A: 仅写到 stderr，保持 stdout 干净。
- Q: 单张图片的高度上限策略？ → A: 宽 ≤ 终端列数且高 ≤ 终端行数的 1/3，等比缩放取较严者。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - View Matching Images Inline In Terminal (Priority: P1)

作为一名在本地或通过 SSH 连接到训练机的用户，我在运行 `aimx query images "<expr>" --repo data` 时，希望除了当前已有的元数据表格/摘要外，`aimx` 能**直接在终端里把匹配到的图片内容渲染出来**，而不是只打印图片的 run/step/name 引用。这样我不必先把图片复制到本地再用看图工具打开，就能肉眼判断训练产物是否符合预期。

**Why this priority**: 这是本次 feature 的核心价值。`aimx query images` 的查询结果（元数据）早已可用；让结果"可视化"是区别于 `aim` 原生 CLI 的关键能力，直接决定本 feature 是否值得存在。

**Independent Test**: 使用仓库根目录下 `data` 中的测试 Aim 仓库，运行 `aimx query images "images" --repo data`；在支持图片协议（如 iTerm2 / Kitty / WezTerm / Ghostty）或回退到 ANSI 半块字符的终端里，可以观察到每一条匹配项下方或旁边出现对应图片的可视渲染。

**Acceptance Scenarios**:

1. **Given** 仓库 `data` 包含至少一条带图片的 run，**When** 用户在支持图形协议的终端中运行 `aimx query images "images" --repo data`，**Then** 终端里每条匹配结果旁边显示出该图片的可视化渲染，且查询元数据（run、step、name、size）仍然可见。
2. **Given** 用户使用不支持任何图形协议的普通终端（仅 ANSI），**When** 运行同一条命令，**Then** `aimx` 使用字符级回退（例如半块字符 / 灰度 ASCII）渲染图片，命令退出码仍为 `0`，不输出 `traceback`。
3. **Given** 匹配结果为空，**When** 运行查询，**Then** 输出与原有 `aimx query images` 的空结果行为一致，不尝试渲染任何图片。

---

### User Story 2 - Suppress Image Rendering When Non-Interactive Or Machine-Readable (Priority: P1)

作为把 `aimx query images` 嵌入脚本或 CI 的用户，我希望在 `--json`、`--plain`/`--oneline` 或 stdout 非 TTY 的情况下，**绝对不要**插入任何用于绘图的转义序列或半块字符，以免污染下游管道/日志。

**Why this priority**: 违反这一点会直接破坏现有使用该命令做自动化的调用方，属于不可回归的契约。

**Independent Test**: 运行 `aimx query images "images" --repo data --json | jq .`、`aimx query images ... --plain > out.txt`、以及把输出重定向到文件，检查产物中不包含任何 ANSI 图形/半块/Kitty/iTerm2 图片协议字节序列。

**Acceptance Scenarios**:

1. **Given** 用户传入 `--json`，**When** 运行查询，**Then** 输出为一个合法 JSON 文档，且**不含**任何图片可视化内容。
2. **Given** 用户传入 `--plain`（或 `--oneline`），**When** 运行查询，**Then** 每行输出保持单行文本、不含图片字节序列。
3. **Given** stdout 被重定向到文件或管道（`isatty()` 为 False），**When** 未显式开启渲染，**Then** 不渲染图片，行为与当前版本保持一致。
4. **Given** 用户传入 `--no-color`，**When** 运行查询，**Then** 继续按 User Story 1 的渲染路径工作，但不使用彩色元数据（是否渲染图片取决于 TTY，不受 `--no-color` 影响）。

---

### User Story 3 - Bounded Output For Large Result Sets (Priority: P2)

作为一次可能匹配到数十/数百条图片的用户，我不希望终端被一次性灌满上千张图片，而是默认只渲染一个可控数量的代表性样本，其余仅保留元数据行并提示"已省略 N 张"。

**Why this priority**: 避免默认行为在大结果集上导致终端卡死或 SSH 断连；是可用性底线，但只要 P1 就绪就可独立交付。

**Independent Test**: 人为构造一个匹配 ≥ 50 条图片的查询，确认默认只渲染前 N 条（N 见 Assumptions），其余只打印元数据并在尾部出现省略提示；通过显式上限标志（见 FR-008）可调整或关闭该上限。

**Acceptance Scenarios**:

1. **Given** 查询匹配到超过默认上限的图片，**When** 按默认参数运行，**Then** 仅前若干条渲染图像，末尾打印形如 `... rendered N of M images, use --max-images=0 for all` 的摘要。
2. **Given** 用户显式传入 `--max-images 0`，**When** 运行查询，**Then** 所有匹配图片全部渲染（用户自担风险）。

---

### Edge Cases

- 图片 blob 缺失 / 损坏 / 解码失败：跳过该条渲染，行内追加简短错误标记（如 `[image unavailable]`），不影响其他条目与退出码。
- 终端宽度极窄（< 20 列）：降级为纯元数据输出，不尝试绘图。
- `textual_image` 或其依赖（PIL 等）运行时不可用：命令仍能完成元数据输出，首行打印一次性警告，后续不再刷屏。
- 非 RGB 图片（灰度、RGBA、P 调色板）：统一转换后渲染，不抛异常。
- 图片尺寸极大（如 >4K）：在渲染前按终端 cell 尺寸缩放到合理宽度（见 FR-006）。
- `--steps` 过滤后剩余 0 条：不渲染，不报错。

## Constitution Alignment *(mandatory)*

- **CA-001 Safety & Mutability**: 本 feature 只新增"展示层"能力，**纯只读**。不写入 `.aim` 数据，不改动已安装的 `aim` 包；仅在 `aimx` 自己的渲染路径中新增输出。因此无需额外的 opt-in 风险提示。
- **CA-002 Ownership Boundary**: `aimx query images` 已归属 `aimx` 实现（见 `src/aimx/commands/query.py`）。本 feature 仍限定在 `aimx` 侧的 rendering 层扩展，不转译或代理任何 `aim` 原生子命令，不改变 passthrough 的边界。
- **CA-003 CLI & Output Contract**: 人类可读模式（默认，TTY）新增内联图片；`--json` 与 `--plain`/`--oneline` 输出契约保持不变（不含图片字节）；非 TTY 默认退回到无图模式。继续满足 shell/SSH/CI 三种使用形态。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 当且仅当 `aimx query images` 在默认（rich）渲染路径下运行、且 stdout 为 TTY 时，系统 MUST 以内联方式在匹配行附近渲染图片内容。
- **FR-002**: 当用户传入 `--json` 或 `--plain`/`--oneline`，或 stdout 非 TTY 时，系统 MUST NOT 输出任何图片字节/图形转义序列，行为与当前 `aimx query images` 保持完全一致。
- **FR-003**: 系统 MUST 优先探测并使用终端原生图片协议（典型如 iTerm2、Kitty、Sixel），在不支持时 MUST 自动回退到纯文本/半块字符渲染，保证在任意 ANSI 终端可用。
- **FR-004**: 图片渲染失败（解码失败、blob 缺失、依赖不可用等）MUST NOT 使整条查询命令失败；受影响的条目以简短占位信息呈现，其余条目照常输出。
- **FR-005**: 系统 MUST 先输出现有的 rich 汇总表格（run、step、context、name、shape 等），随后在表格下方按每条匹配项分段打印"元数据头 + 对应图片"块。图片渲染不得改变或覆盖汇总表格的可见性。
- **FR-006**: 系统 MUST 在渲染前按图片原始宽高比**等比缩放**，同时满足以下两个上限，取较严者：(a) 目标宽度 ≤ 当前终端列数；(b) 目标高度 ≤ 当前终端行数的 1/3（按字符 cell 估算）。避免任何单张图片独占一屏或触发横向换行撕裂。
- **FR-007**: 默认行为 MUST 对大结果集做数量上限保护：匹配图片数超过默认上限（**6 张**）时，只渲染前 6 张，其余以元数据行呈现，并在结尾追加一次性摘要提示（形如 `... rendered 6 of M images, use --max-images=0 for all`）。
- **FR-008**: 系统 MUST 提供一个命令行开关 `--max-images N` 以调整 FR-007 的上限（`N=0` 表示不限）。系统 MUST NOT 引入专门用于关闭内联渲染的新开关；用户若要在 TTY 下完全不渲染图片，应使用既有的 `--plain` / `--json`，或将 stdout 重定向到管道/文件。
- **FR-009**: 系统 MUST 保证在依赖 `textual_image` 或其运行时依赖缺失的环境下，命令主流程（元数据输出与退出码）不受影响；缺失信息以**一次性 warning 写到 stderr**（不进入 stdout），同一进程内不重复打印。

### Key Entities *(include if feature involves data)*

- **Image Query Row**: 既有实体，新增一个可选的"已解码图片位图引用"字段，供渲染层消费；该字段不进入 `--json` 输出。
- **Terminal Capability**: 描述当前 stdout 对应终端的绘图能力（协议支持、列/行数），在命令启动时一次性探测，供后续分支决策使用。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 在支持图形协议的现代终端（iTerm2 / Kitty / WezTerm / Ghostty 任一）中，用户首次运行 `aimx query images "images" --repo data` 时，**无需任何额外参数**即可看到图片内容；达标率 100%（该类终端上）。
- **SC-002**: 在只支持 ANSI 的通用终端（tmux 默认配置、常规 ssh-only 场景）中，同一命令 MUST 在不报错的前提下输出字符级图像回退，命令退出码为 `0`。
- **SC-003**: `aimx query images ... --json` 的输出字节与本 feature 引入前的输出字节**逐字节一致**；同样，`--plain` 与重定向到文件的输出不含任何新增的控制序列。
- **SC-004**: 对匹配 ≥ 50 张图片的仓库，默认命令在普通开发机上 5 秒内返回完毕并停止继续渲染（默认上限生效）。
- **SC-005**: 在缺失 `textual_image` / PIL 依赖的环境，命令仍能输出元数据表格、退出码为 `0`，并在顶部打印一条关于"内联图片渲染不可用"的警告。

## Assumptions

- 默认匹配上限由 FR-007 固定为 6 张；可通过 `--max-images` 覆盖（`0` 表示不限）。
- 渲染实现基于已在 User Query 中提出的 `rich.Console + textual_image.renderable.Image` 组合；`textual_image` 将作为 `aimx` 的新运行时依赖加入 `pyproject.toml`（实现阶段处理，不在本 spec 范围）。
- 终端能力探测仅基于环境变量（`TERM`、`TERM_PROGRAM`、`KITTY_WINDOW_ID` 等）和 `stdout.isatty()`，不做主动的 DA/Device Attribute 查询（避免阻塞 SSH/CI 场景）。
- 图片 blob 的加载仍沿用既有 `aimx.aim_bridge` 的只读访问路径，不新增对 Aim 内部 API 的写入或变更。
- 首版只覆盖**单帧静态图片**；多帧序列 / GIF / 视频渲染不在本 feature 范围。
- 颜色深度以终端支持的最大真彩色为准；`--no-color` 仅影响元数据样式，不强制禁用图片渲染。
