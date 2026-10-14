# Implementation Plan: Autoresearch 研究成果沉淀与可靠交接

**Branch**: `main`（实际 Git 分支） | **Feature ID**: `008-autoresearch-knowledge-loop`（setup 的 BRANCH 值） | **Date**: 2026-10-13 | **Spec**: [spec.md](spec.md)

**Input**: `specs/008-autoresearch-knowledge-loop/spec.md`。本次按 spec 的目录/分支独立约定在 main 规划，未创建功能分支。

## Summary

扩展已有 Research State 和仓库内 autoresearch skill，解决“有成果但找不到、文件证据无法入 Finding、上下文漏修正、失败交接失真”的具体问题。P1 提供 loop 概览/回执、层级帮助、文件证据和 Finding 语义、确定性完整修正上下文；P2 提供终态后独立补证活动、实验合同、数据观察记录、多轴 attempt 和预检/失败策略。五个 user stories 全部在本期设计范围，可依 P1→P2 分阶段交付，不把 P2 当可忽略项。

保留 Python CLI 与本地 SQLite append-only commits/events，复用现有 optimistic revision 和请求指纹幂等机制。ResearchUpdate 在现有 schema_version=1 上增量扩展可选字段和新事件，继续接受旧载荷与历史；Receipt 是 checkpoint 与知识事实的只读聚合，不能成为第二份可变状态。Aim run 继续只读验证，新增本地普通文件的 SHA-256 身份；科学有效性独立审核。context 统一计整个响应预算，递归包含已知必要修正；命令用法保持，不提供格式选择或旧行为分支。

规划阶段已结束；实现和当前验收证据见 [tasks.md](tasks.md) 与 [validation.md](validation.md)。

## Technical Context

**Language/Version**: 开发 Python 3.12；运行兼容 `>=3.10,<3.13`。

**Primary Dependencies**: 标准库 sqlite3/json/pathlib/hashlib/dataclasses/uuid、现有 Rich；既有 Aim SDK 仅用于 Aim 引用，来自 dev group/外部安装，不成为读取文件或纯 sidecar 的必需条件。numpy/plotext/textual-image 的既有功能保留；无新增运行时包、在线 LLM、向量库或服务。

**Storage**: `.aimx/research/state.sqlite3` 现有 commits/events/metadata；`.aimx/autoresearch/<loop_id>/state.json` 保存派发/预算/终态事实；外部报告与证据只引用、不搬运。同一 ResearchUpdate/回放路径增量扩展，schema_version 仍 1，读不 ALTER/迁移；checkpoint 结构和配置指纹保持。

**Testing**: pytest unit/contract/integration，uv 管理 `.venv`；模拟宿主/evaluator 与可移植 temp repo；保留 10,000 commits / 50,000 events scale fixture、真实 `data/.aim` 只读 smoke 和 passthrough 回归。

**Target Platform**: 本地终端、SSH、CI；优先 Linux、兼容现有 Python 支持平台，不依赖 GUI、GPU 或在线模型服务。

**Project Type**: 单项目 companion CLI + 可安装 agent skill；宿主负责派发/模型，项目负责实验/evaluator/计算约束。

**Performance Goals**: SC-001 五类成果样例最多三次查看、两分钟内找到状态/位置/结论/入口；SC-004 十轮 fixture 的 18,000-byte 响应包含最终认识及必要修正；更小预算完整或明确不足。沿用研究规模下本地交互目标约两秒（排除 Aim lookup/显式文件哈希），以既有 scale fixture 检查复杂度，不能靠截断修正达标。

**Constraints**: `.aim` 和 installed Aim 只读；查询/帮助/证据检查不创建 sidecar、不升级 checkpoint；显式写仅 Aimx owned sidecar；旧 ID/记录/Aim refs/原命令/退出码保持；unknown root 透传 fidelity。历史无活动归属不猜测；未知观察状态不能声称独立；没有执行器能力就不承诺计算预算强制。

**Scale/Scope**: 5 个 user stories、FR-001–FR-021、SC-001–SC-008；同 repo 多 loop 和跨活动关系，成千 Finding 与数万事件。远程抓取/同步、全局 repo 索引、自动搬运、权限系统、训练/GPU 调度、进程取消、硬预算平台不在范围。

初始待研究的兼容扩展、文件身份、上下文预算、提交/loop 协调、项目实验能力五组未知项已全部在 [research.md](research.md) 收敛。项目特定阈值/能力是后续显式配置输入，缺项按 preflight 阻塞，不是规划悬而未决的选择。

## Constitution Check

*GATE: Phase 0 前已检查；Phase 1 后复检如下。*

根目录 `CONSTITUTION.md` v1.0.0 为仓库现存已填充治理文本，`.specify/memory/constitution.md` 已读取但仍是未填充模板，两份镜像声明不一致。按 spec 明确约定采用现行根治理原则，不把模板当批准规则，本次不修改治理文本。该现存文档问题不造成设计原则例外；下表对实际设计逐项检验。

| Gate | Phase 0 前 | Phase 1 后 | 设计/验证依据 |
| --- | --- | --- | --- |
| I 安全共存 | PASS | PASS | Aim ref 只读桥接；文件只读，新增写仅 `.aimx`；snapshot 验证 `.aim` 不变 |
| II 明确所有权/透传 | PASS | PASS | 仅已有 research/finding/lineage/frontier 树；native help/argv/stdout/stderr/exit 保留回归 |
| III CLI/可脚本化 | PASS | PASS | 新入口 human/JSON，无 repo help、stdin/file update，统一紧凑 context |
| IV 默认只读/安全失败 | PASS | PASS | help 在 IO 前；overview/check/receipt 不写；显式 update；不自动提炼/修复 |
| V 聚焦扩展 | PASS | PASS | 为 Aim 用户沉淀成果；项目/宿主能力合同，无训练服务或调度平台 |
| uv 与兼容 gate | PASS | PASS | 无新 runtime 包；旧 input/events/checkpoint 可读，无格式升级流程；未知事件安全报错 |

无未获准的原则违反，Complexity Tracking 无例外。兼容承诺是新程序读取旧数据、旧载荷继续有效，不要求旧程序理解新增事件。上下文预算从 items-only 收紧到完整响应是明确的行为变化，README/release 和测试需同步说明，不以版本开关维持双轨。

## Project Structure

### Documentation (this feature)

```text
specs/008-autoresearch-knowledge-loop/
├── spec.md                         # 原规格，保持原文
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli-output.md
│   ├── research-update.md
│   ├── context.md
│   └── autoresearch-protocol.md
└── checklists/requirements.md       # 原检查单
```

`tasks.md` 留给后续 /speckit.tasks，本阶段不创建。

### Source Code (repository root)

```text
src/aimx/
├── router.py                       # 原 ownership 保持，验证 help/passthrough
├── cli.py                          # owned dispatch 保持
├── commands/
│   ├── research.py                 # overview/annotations/receipt/evidence + 统一 context
│   ├── finding.py                  # topic/loop 查询与无 repo help
│   ├── lineage.py                  # 关系展示与层级 help
│   ├── frontier.py                 # 层级 help，长期方向与 loop 状态区分
│   └── help.py                     # 静态命令 help 定义
├── research/
│   ├── models.py                   # 新实体/来源与 evidence union
│   ├── ids.py                      # 新 entity prefixes（复用现有机制）
│   ├── events.py                   # 增量 operation catalogue
│   ├── validation.py               # 可选扩展字段、来源/观察/合同校验
│   ├── replay.py                   # 混合版本只读投影
│   ├── store.py                    # 原子追加/幂等/只读 receipt 查询
│   ├── context.py                  # lexical selection / recursive closure / full budget
│   ├── agenda.py                   # 保持现有调度建议顺序，不替代宿主
│   ├── evidence.py                 # 拟新增：本地文件身份、检查、kind 分派
│   └── overview.py                 # 拟新增：checkpoint + state 的只读聚合
├── aim_bridge/research_evidence.py # 仅 Aim 引用，区分 SDK/缺 run
└── rendering/research_views.py     # 人类视图；统一紧凑 context JSON

skills/aimx-autoresearch/
├── SKILL.md                        # 实际预检、合同、失败策略、成果回执
├── scripts/control.py              # 保持 checkpoint/config/fingerprint，复用现有入口
├── references/loop-contract.md
├── references/research-handoff.md
└── assets/config.example.json      # 保留现有配置形状；项目合同另行示例

skills/aimx/
└── references/autoresearch-protocol.md # 同步读取/证据指引（既有文件）

README.md                           # 所有权、预算语义变化、回执与验证流程（实施阶段）
tests/
├── unit/test_research_*.py          # 复用已有 store/replay/context/evidence/scale 等
├── unit/test_autoresearch_control.py
├── contract/test_research_*.py
├── contract/test_autoresearch_protocol.py
├── integration/test_research_commands.py
├── integration/test_autoresearch_skill_loop.py
├── integration/test_passthrough_behavior.py
└── fixtures/research_knowledge_loop/ # 拟新增可移植十轮/多 loop/故障样例
```

**Structure Decision**: 延续单一 Python CLI 的模型/校验/存储/渲染边界，仅增加两个职责明确模块：evidence 隔离普通文件 IO 与 Aim bridge，overview 隔离 checkpoint 读取与只读聚合。实验协议继续放仓库安装源，不增加独立服务/数据库/通用执行框架。新增目录和模块均标明拟新增。

## Phase 0: Research Decisions

细节、备选方案和具体源码符号见 [research.md](research.md)。两名 Luna worker 完成 state/interfaces 与 loop/protocol 的只读调查，结果已由父会话消费；最终决策由父会话形成：

- 单一 update/回放路径增量扩展，schema_version 保持 1；旧载荷可用，新增语义提供时严格校验，不迁移旧事件。
- Activity 作归属锚点；checkpoint/知识分别保留事实，Receipt 不持久化独立状态。
- 普通文件 SHA-256 身份 + 来源/范围/限制；可用性、身份和科学审核分开。
- context 先真实文本匹配，再完整递归必要修正，实际响应含 metadata/refs 全计预算；现有调用方已核对，统一更新预算语义，无用户版本参数。
- 稳定 key/请求指纹/原文件协调；终态后补证采用独立活动，不追加原第 11 轮。
- 严格实验合同和数据观察事件；实际 preflight 与声明计算预算，宿主仍掌控执行。

## Phase 1: Design & Contracts

### P1 可发现性、证据与上下文

1. 静态层级 help 在 repo/ID/SDK/store IO 前返回；新增 overview、annotations、receipt、evidence check 和 ls 的 topic/loop filter，全部非交互 human/JSON。无 loop 的概览列表为登记 loop 提供 Finding/annotation 计数及 Finding 摘要，方便区分非 Finding 历史成果。
2. 从原 append-only history 投影 Activity、文件 Evidence、Finding result/scope/limitations/topics；来源缺失为 unknown/unattributed。显式 source_annotation_ids 记录人工提炼来源，不在读时转换。
3. Receipt 聚合实际 repo/state/checkpoint/report、原 goal/rounds/stop、知识计数/摘要、可核对 commit、未协调事项；合法空与不可读严格区分。
4. context 递归 challenges/refines/supersedes 闭包，保证选中的结论与重要修正一起出现；包括跨活动关系，保持确定性。BudgetError 不在 stdout 输出单边结论。annotation-only 相关性选择承认文本关系不可靠。
5. receipt 查询复用已存 result/fingerprint；原 exact update 精确重放仍可协调；保存产物不能等同 commit 成功。

### P2 补证与可靠实验协议

1. 原终态 checkpoint/预算/标准不变，新增 follow_up Activity、Finding、关系与回执区段；原历史结果与后续认识并列。
2. Agenda 的可选结构化 experiment_contract 冻结问题/干预/对照/混杂/反证/评估/数据范围/候选/资源/失败策略；项目 evaluator 给出真实预检与评分协议。
3. DatasetScope/Observation 记录声明、冻结、首次评分、转开发，含来源和内容身份；未知/观察/重叠数据拒绝独立确认声明，允许诚实探索；记录外读取限制明确可见。
4. Attempt 分开执行、产物、评估、审计、实际设置、预算计数。executor 重试记录不能授予 assistant 重放 slot 的权限；没有日志/约束标未核验/未强制。
5. helper 沿用同一 checkpoint/config/result 形状与指纹算法，无 upgrade；预检/合同/Attempt 四轴通过关联研究记录保存，继续旧循环只检查具体缺项。先预检/合同/消耗轮次，再 reserved→spawn→started；策略未验证停止后续派发。

详尽接口以 [CLI](contracts/cli-output.md)、[Update](contracts/research-update.md)、[Context](contracts/context.md)、[Skill protocol](contracts/autoresearch-protocol.md) 为规范；实体关系与转换见 [data-model.md](data-model.md)，验证入口见 [quickstart.md](quickstart.md)。合同例子是未来行为，不能当当前已实现结果。

## Requirements / Validation Coverage

| 要求 | 设计位置 | 验证场景 |
| --- | --- | --- |
| FR-001 / FR-002 | overview/Receipt；skill finish | 成功/未达标/阻塞/annotation-only/多 loop，提交不确定回执，SC-001/008 |
| FR-003 | 各层静态 help | 所有父/叶 -h/--help，无 repo/ID，禁止 IO，SC-002 |
| FR-004 | ls other counts；annotations | 合法空、历史 annotation、未知/坏 repo，SC-001/006 |
| FR-005 / FR-006 / FR-007 | Evidence union/Finding/scope/review | 四类 result、文件失踪/权限/digest 改变与科学状态分离，SC-003/006 |
| FR-008 | topic/loop/source_annotation_ids/show | 不匹配/归属/人工提炼来源/关联范围，SC-001/003/005 |
| FR-009 / FR-010 / FR-011 | 统一 context | 十轮 18,000-byte、递归/环/跨活动、N/N-1、annotation-only、遗漏/确定性，SC-004 |
| FR-012 | follow_up/new Finding/relations | 原终态 bytes/轮数/criteria 不变；一次 topic 查询包含限制，SC-005 |
| FR-013 | append-only/idempotency/receipt | 同 key 同请求/不同请求、并发、冲突、提交中断、旧不可核验回执，SC-008 |
| FR-014 / FR-015 | contract/preflight/实际命令检查 | 缺字段/入口/模型/evaluator/Aim 能力，未验证不 spawn，SC-008 |
| FR-016 | scope/observation/frozen candidate | 已评分、开发、未知/重叠、冻结次序和允许探索，SC-007 |
| FR-017 / FR-018 | Attempt 四轴/resource facts | 服务失败完整预测/部分产物/多 attempt/计数缺失与超额，SC-008 |
| FR-019 | failure_policy/host/no-replay | stop_pending、已验证 continue、未知策略、固定模型/槽位/轮数 |
| FR-020 / FR-021 | ownership/read vs explicit write | before/after 文件快照、旧载荷/events/Aim ref、human/JSON、native exit fidelity，SC-006 |

实施时要求每个 SC 都有具体 fixture 和证据；SC-001 是定时可发现性验收，不能仅以单元测试替代。所有 fixtures 可移植，不依赖真实 ICL 路径、GPU、在线模型或用户历史状态。完整命令见 quickstart。

## Post-Design Constitution Check

- PASS：所有 read/help/check/context/receipt 只读，无自动迁移、提炼、修复或目录创建。
- PASS：显式 update 的唯一产品持久写表面为 Aimx sidecar；不修改 Aim、用户源码/生产配置。
- PASS：原所有权/透传不扩大，human/JSON/退出码有规范。
- PASS：新程序接受旧输入和事件；新字段可选兼容，008 工作流单独检查完整性；checkpoint 不转换格式，缺关联合同/预检阻塞派发，终态不可重开。
- PASS：独立性/硬预算只声明已核验能力，计算/训练/取消由项目或宿主提供，未引入平台。
- PASS：无新增包，开发/验收使用 uv；未来修改函数必须先 GitNexus impact，未来提交前 detect_changes。本阶段只创建规划文档，不修改程序符号或提交。

## Planning Validation & Hooks

Phase 1 文档在用户确认单一 CLI 后已重新校验：8 份产物的相对链接有效，9 个 JSON 示例可解析且 schema_version 均为 1，quickstart 8 段 bash 语法有效，临时样例 update 的真实 SHA-256/size 与证据字段一致；FR-001–FR-021 与 SC-001–SC-008 全部有覆盖映射。已检查无双版本参数、版本协商字段或 checkpoint 格式升级设计残留；仅保留内部 compiler_version 算法标记。没有未决澄清/模板占位，未生成 tasks.md。`git diff --check` 通过。

前置和后置均检查 `.specify/extensions.yml`：文件不存在，before_plan/after_plan 按规则跳过，无待执行必需钩子。功能 pytest 未运行，因为本轮仅规划；尝试 `uv run python --version` 时触发环境同步下载并在 20 秒超时，未得到成功的运行环境验证。文档检查使用现有解释器，不把它当实现或环境就绪证明。工作区原有 AGENTS.md/.claude/.pi 改动保留，未提交。

## Complexity Tracking

无宪法违反，无需例外。活动、观察、attempt 复用一套已有事件/事务机制；不建新数据库。递归闭包及完整响应字节计数是 SC-004 的必要复杂度；用户已确认移除双版本选择、版本协商和 checkpoint 升级。内部 compiler_version 只用于追溯算法，不维护旧编译器。普通文件/hash 和 checkpoint 读取分别隔离到 evidence/overview，避免命令层聚集领域逻辑。
