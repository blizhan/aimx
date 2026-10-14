# Tasks: Autoresearch 研究成果沉淀与可靠交接

**Input**: `specs/008-autoresearch-knowledge-loop/` 的 [plan.md](plan.md)、[spec.md](spec.md)、[research.md](research.md)、[data-model.md](data-model.md)、[contracts/cli-output.md](contracts/cli-output.md)、[contracts/research-update.md](contracts/research-update.md)、[contracts/context.md](contracts/context.md)、[contracts/autoresearch-protocol.md](contracts/autoresearch-protocol.md)、[quickstart.md](quickstart.md)。

**Tests**: spec 的 Acceptance Scenarios、SC-002/004/006/007/008 明确要求验收覆盖与测试样例，故每个故事包含有针对性的测试任务；先编写并确认新增断言因预期行为未实现而失败，再实施，不引入额外用户审批步骤。

**Organization**: Setup → Foundation → 五个按优先级排列的用户故事 → 跨切验收。任务均未执行；生成清单不等于功能已实现。

## Format / Paths / Working Agreement

每个任务使用 `- [ ] Tnnn [P?] [USn?] 描述 + 文件路径`。路径相对仓库根；实施时实际文件操作使用绝对路径。`[P]` 仅指下文明确 ready wave 内可以并行，不授权自动派生 agent；同一 cwd 只有一个 writer，多 writer 必须隔离并串行集成。

- Python 开发使用 3.12、`uv sync`/`uv run`、项目 `.venv`；无新增运行时依赖。
- 单一 CLI，schema_version 保持 1；新增可选字段/事件，不增加版本选择、版本协商命令或 checkpoint 升级流程。compiler_version 仅标算法。
- 既有 update 必需字段、请求 fingerprint、config 规范化/fingerprint、checkpoint/finished.result 形状不变。008 工作流在派发/成果提交时检查完整性；旧输入缺新字段继续有效。
- 读/help/check/context/receipt/dry-run 不创建或修复 `.aimx`，不写 `.aim`、installed Aim、项目源码或生产配置；写入仅明确的 Aimx sidecar update/checkpoint。
- 修改任何函数/类/方法前，按 `AGENTS.md` 运行 GitNexus upstream impact 并报告直接调用者、流程和风险；HIGH/CRITICAL 必须先警示。每个后续符号修改仍需对应 impact，不能以一次基线查询替代；提交前 detect_changes。
- `.specify/memory/constitution.md` 是占位模板；沿用 spec/plan 已记录的根 `CONSTITUTION.md` 原则，不在本清单加入治理修订。
- 验证使用临时 repo、可移植 fixture 和假宿主/evaluator；私有 ICL 报告仅作问题来源。真实 `data/.aim` 仅只读 smoke。失败/未运行必须如实记入 `specs/008-autoresearch-knowledge-loop/validation.md`（实施时新建）。

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 确认现有工程与验证入口，建立可移植输入和兼容性基线，不重新初始化项目。

- [x] T001 在仓库根使用 uv sync --python 3.12 和 uv run python --version 确认 pyproject.toml 对应 .venv，运行 quickstart.md 的既有基线命令并在 specs/008-autoresearch-knowledge-loop/validation.md 记录版本、结果及环境阻塞；不更改依赖声明。
- [x] T002 核对 AGENTS.md、CONSTITUTION.md 与 specs/008-autoresearch-knowledge-loop/contracts/ 的单一 CLI 约束，在 specs/008-autoresearch-knowledge-loop/validation.md 建立 FR/SC 证据表、GitNexus 每次 impact/提交前 detect_changes 的记录位置和 fixture 使用规则。
- [x] T003 [P] 在 tests/fixtures/research_knowledge_loop/ 新建 README.md、legacy-state.json、loop-receipts.json 和 ten-round-history.json，可移植描述五类回执、多 loop/未归属记录、十轮二十条 annotation、早期增益与第九/十轮修正；实际 repo/产物路径及真实 digest 由测试 temp dir 生成。
- [x] T004 [P] 在 tests/contract/test_research_schema.py 添加旧 envelope/原 operation/最简 Aim ref 继续有效、未知字段拒绝及现有 schema 与 operation catalogue 一致性的基线；新增断言只覆盖本期可选 provenance 扩展，不要求任何用户版本参数。

**Ready waves**: T001 → T002 → T003 与 T004。fixture 为描述输入，不把历史 annotation 自动转 Finding。

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 所有故事共用的增量兼容、来源保存和原子/幂等边界。此阶段完成前不开始故事实施。

- [x] T005 在 src/aimx/research/models.py、src/aimx/research/validation.py 和 src/aimx/research/replay.py 增量允许既有 operation 的可选 source_activity_id，保留旧必需字段/作者/提交历史；缺来源仅在读取投影中显示未归属，不注入原请求或补写历史，实体引用校验在 US1 活动接入后补齐。
- [x] T006 在 tests/unit/test_research_store.py 验证旧请求 fingerprint、相同 key 精确重放、多操作回滚、revision conflict 和旧库只读不 ALTER；核对 src/aimx/research/store.py 的既有路径，必要修复不得改变 canonical 请求算法或向幂等请求注入扩展默认值。
- [x] T007 在 specs/007-research-state/contracts/research-update.schema.json 和 tests/contract/test_research_schema.py 同步可选 provenance 及严格键规则，保持 schema_version=1 与原必需字段；为以后每个故事追加 operation 的同步规则保留同一 schema 文件，不另建版本 schema。
- [x] T008 运行 tests/contract/test_research_schema.py、tests/unit/test_research_store.py、tests/unit/test_research_validation.py 和 tests/unit/test_research_replay.py 的 uv run pytest，确认旧载荷/旧回放/原子性基线通过并记录 specs/008-autoresearch-knowledge-loop/validation.md。

**Checkpoint**: Foundation ready；仅有公共来源/兼容机制，具体活动、证据和实验实体在对应故事实现。

## Phase 3: User Story 1 — 找到一次研究留下的成果 (P1, MVP)

**Goal**: 成果概览、非 Finding 入口、无需 repo 的层级帮助和所有停止路径的诚实回执。

**Independent Test**: 五类临时样例（成功、未达标、阻塞、仅历史 annotation、多 loop）中最多三次查看、两分钟内找到实际 repo/状态/成果/证据入口；所有父/叶帮助无 IO；坏状态输出 partial 而非空成果。无需文件 Finding 或新实验协议。

### Tests

- [x] T009 [P] [US1] 在 tests/contract/test_research_contract.py 参数化 contracts/cli-output.md 中全部现有及新增父/叶 -h/--help 与 JSON help，断言缺 repo/实体 ID 仍 exit 0、用途/参数/示例/读写/退出码齐全，并禁止 repo/store/Aim 访问；补充 overview/annotations/receipt 的字段和退出码断言。
- [x] T010 [P] [US1] 在 tests/integration/test_research_commands.py 消费五类回执 fixture，覆盖 repo 根/.aim/外部 cwd、多 loop 不串数据、旧记录未归属、Finding 空列表提示、checkpoint 缺失/损坏、DB 不可读和保存产物但提交不确定，检查前后文件 bytes 不变。

### Implementation

- [x] T011 [US1] 在 src/aimx/research/models.py 和 src/aimx/research/ids.py 定义 Activity 的 act_ 身份及 loop/analysis 来源、导航位置和只读投影；保留旧实体序列化字段，缺归属不推断，follow_up 语义留 US4。
- [x] T012 [US1] 在 src/aimx/research/events.py、src/aimx/research/validation.py、src/aimx/research/replay.py 和 specs/007-research-state/contracts/research-update.schema.json 实现 activity.create/activity.artifact.add 的 loop/analysis 分支、同 update 引用、可选来源实体校验与追加导航；loop 登记校验标准 checkpoint 的 repo/ID，不改 checkpoint。
- [x] T013 [US1] 在 src/aimx/research/store.py 增加按 client_update_id 与可选 fingerprint 的只读 receipt 查询，返回原 result_json 或 not_found/idempotency_conflict/receipt_unverifiable，保留查询快照与并发限制，不初始化或 ALTER 数据库。
- [x] T014 [US1] 新建 src/aimx/research/overview.py，只读发现标准 checkpoint 并聚合知识快照、原 goal/轮次/stop、方向状态、实际路径、commit 回执和产物导航；无 loop 返回列表，指定未知 loop 报错，坏来源保留安全可读信息并标 partial/待协调，不持久化 Receipt。
- [x] T015 [US1] 在 src/aimx/commands/help.py 定义 research/finding/lineage/frontier 每个层级的静态 text/JSON 帮助，覆盖新增 overview/annotations/receipt/evidence check 的语法与性质；仅登记帮助不开放尚未实现的业务入口。
- [x] T016 [US1] 在 src/aimx/commands/research.py、src/aimx/commands/finding.py、src/aimx/commands/lineage.py、src/aimx/commands/frontier.py 及必要的 src/aimx/cli.py 分派处，在 repo/ID/SDK IO 前处理已识别层级 help；未知子命令仍 exit 2，不拦截 native Aim 子命令帮助。
- [x] T017 [US1] 在 src/aimx/commands/research.py 接入 overview、annotations（可选 topic/loop）和 receipt 命令，严格区分合法空存储/未知路径/坏状态，按 contracts/cli-output.md 输出 human/JSON、stdout/stderr 和 exit 0/2/3。
- [x] T018 [US1] 在 src/aimx/rendering/research_views.py 和 src/aimx/commands/finding.py 实现 Receipt human/JSON、空 Finding 的其它记录计数/未归属提示及可复制查询；查询命令正确引用含空格路径，goal/loop/方向三状态分离。
- [x] T019 [US1] 更新 skills/aimx-autoresearch/SKILL.md 和 skills/aimx-autoresearch/references/research-handoff.md：lead 显式登记活动/来源，保存 exact update/key/digest，结束/暂停/阻塞/失败均使用 overview 交接；产物、持久化知识、待协调分列，checkpoint/config 形状不变。
- [x] T020 [US1] 运行 T009/T010 及 tests/integration/test_passthrough_behavior.py，按 quickstart.md 在五类 fixture 上完成 SC-001 三次操作/两分钟计时检查和 SC-002 全层帮助枚举，把实际命令/计时/缺项记录 specs/008-autoresearch-knowledge-loop/validation.md。

**Checkpoint**: US1 可独立交付；旧 annotation 已可发现但不自动转换，后续故事缺失不妨碍本故事验收。

## Phase 4: User Story 2 — 将负结果和文件证据保存为可复用发现 (P1)

**Goal**: 文件证据支持四类 Finding 语义，检索与详情可追溯，身份/可用性/科学审核分开。

**Independent Test**: 无 Aim run/无训练提交临时评估产物 Finding，topic/loop 查询及 show/check 可追到来源、范围、限制；四类结果和文件变动诊断正确；显式提炼保留 annotation 来源。执行/科学/goal 的不同含义先用静态回执表达，完整 Attempt 持久化在 US5。

### Tests

- [x] T021 [P] [US2] 在 tests/unit/test_research_evidence.py 添加普通文件流式 SHA-256/size、相对 repo/绝对外部路径、symlink、特殊文件拒绝、读取中变化、缺失/模拟权限错误/内容变化和无 Aim SDK 的测试，断言检查不修改原引用或 revision。
- [x] T022 [P] [US2] 在 tests/unit/test_research_validation.py 添加四类 result_kind、范围/限制/来源成组完整性、旧 Finding/Aim ref 仍有效、显式 source_annotation_ids、evidence.review 对应引用/依据及失败整单回滚测试。
- [x] T023 [P] [US2] 在 tests/integration/test_research_commands.py 添加文件 Finding 提交/检索/show/check、零匹配不混入治理加权项、删除/修改后保留历史、区间跨零保持 candidate 和提炼不自动执行的验收场景。

### Implementation

- [x] T024 [US2] 在 src/aimx/research/models.py 扩展 EvidenceReference 的 aim_run/local_artifact 联合、Finding 的可选 topics/result_kind/scope/limitations/source_annotation_ids 和 review 投影；旧字段/枚举原语义保留，未知与未审核不伪装成否定。
- [x] T025 [US2] 新建 src/aimx/research/evidence.py，实现普通文件登记时 digest/size 校验、原位置/解析位置保存、symlink/读前后身份检查及只读 EvidenceCheck；availability/identity/review 分轴，不复制、搬运、执行文件或自动修正 digest。
- [x] T026 [US2] 在 src/aimx/aim_bridge/research_evidence.py 与 src/aimx/commands/research.py 按 evidence kind 分派和按需访问 Aim，保留 full hash/歧义校验，区分缺 SDK 与缺 run；纯文件/sidecar/help 不依赖 Aim 可读。
- [x] T027 [US2] 在 src/aimx/research/validation.py、src/aimx/research/events.py、src/aimx/research/replay.py 和 specs/007-research-state/contracts/research-update.schema.json 实现文件引用、Finding 完整性、显式 annotation 来源及 evidence.review 的追加历史；不新增版本分支、不改变原请求 fingerprint。
- [x] T028 [US2] 在 src/aimx/commands/finding.py 增加 ls 的 topic/loop 过滤及匹配理由，show 返回相关 Finding 的范围/来源/证据与修正；旧无过滤排序保留，loop 来源过滤不冒认 follow_up 或未归属记录。
- [x] T029 [US2] 在 src/aimx/commands/research.py 和 src/aimx/rendering/research_views.py 接入 evidence check 的 human/JSON 及 Finding 四类语义/三种证据状态轴显示；异常引用是成功诊断，未知实体/坏状态 exit 2，不把文件 matches 当科学 accepted。
- [x] T030 [US2] 更新 skills/aimx-autoresearch/references/research-handoff.md 和 skills/aimx/references/autoresearch-protocol.md，使用真实文件身份提交完整成果或显式提炼，不再要求非 Aim 证据只能放 annotation；无可靠主张允许仅保留 annotation/产物。
- [x] T031 [US2] 运行 T021–T023、tests/contract/test_research_schema.py 及 quickstart.md 的文件 payload/dry-run/commit/check/replay 流程，验证 SC-003/SC-006 四类结果和只读边界，记录 specs/008-autoresearch-knowledge-loop/validation.md。

**Checkpoint**: US2 可在静态评估产物上验收，不依赖新上下文选择或真实实验执行。

## Phase 5: User Story 3 — 后续研究获得包含修正的上下文 (P1)

**Goal**: 有限预算上下文包含相关结论和必要修正，结构化证据、遗漏原因及历史文本限制可见。

**Independent Test**: 十轮 fixture 在 18,000-byte 完整 JSON 响应中包含最终解释；多跳修正/环/跨活动不能拆组；N/N-1 或极小预算完整或明确失败，纯 annotation 最新相关结论可见，固定输入重复选择一致。

### Tests

- [x] T032 [P] [US3] 在 tests/unit/test_research_context.py 添加英文/中文真实命中、多跳 challenges/refines/supersedes 闭包与环、跨 loop 反证、annotation-only 修正/关系限制、混合证据去重用途和大量遗漏计数测试，禁止未命中但 accepted 的 Finding 充当 seed。
- [x] T033 [P] [US3] 在 tests/contract/test_research_contract.py 添加整个 stdout UTF-8 字节数（含 LF/metadata/refs/notes/budget_used 固定点）、N/N-1/tiny budget、stderr 诊断与 Human 预算作用域合同测试，更新旧 items-only 断言且不保留格式选择分支。

### Implementation

- [x] T034 [US3] 在 src/aimx/research/context.py 实现归一化词法命中、明确相关性门槛和稳定排序，并将 active challenges/refines/supersedes 求递归无向闭包；loop 仅过滤 seed，跨活动必要反证标 closure_dependency，visited 防环不使用 depth 截断。
- [x] T035 [US3] 在 src/aimx/research/context.py 实现 annotation 按相关性/后续 revision 选择、明确实体 target 的解释组、无法证明自由文本关系的限制、混合结构化 evidence_refs 与用途映射及 selected/omitted 理由；普通文本路径不得升级为证据。
- [x] T036 [US3] 在 src/aimx/research/context.py、src/aimx/rendering/research_views.py 和 src/aimx/commands/research.py 统一完整响应预算与紧凑稳定 JSON+LF 序列化，将 budget_used 迭代至实际字节固定点；最相关必要组/最小 envelope 不适配时 stdout 为空并给 required_bytes，Human 明确 JSON 预算作用域。
- [x] T037 [US3] 在 src/aimx/commands/research.py 接入 context --loop，保持原 objective/budget/json 调用方式、schema_version=1 和内部 compiler_version=context-v2；无用户版本开关、无旧编译器回退，选择不受实时检查时钟影响。
- [x] T038 [US3] 更新 tests/unit/test_research_scale.py 的旧字节断言并扩展共享结构索引的规模验证，保持 10,000 commits/50,000 events 和大量 Finding 场景，检查递归闭包无重复/全图逐项扫描退化；耗时指标排除 Aim lookup 与显式文件哈希。
- [x] T039 [US3] 更新 skills/aimx/references/autoresearch-protocol.md 和 skills/aimx-autoresearch/references/research-handoff.md 的上下文调用/预算说明：使用完整响应预算，budget 不足报告完整查询入口，不截掉反证或静默回退算法。
- [x] T040 [US3] 运行 T032/T033、tests/unit/test_research_scale.py 和 quickstart.md 的十轮 18,000-byte、N/N-1、重复输出检查，记录 SC-004 的完整闭包/明确不足证据及预算语义变化于 specs/008-autoresearch-knowledge-loop/validation.md。

**Checkpoint**: US3 不要求真实后续提交；fixture 中的跨活动关系足以独立验证修正完整性。

## Phase 6: User Story 4 — 循环结束后继续修正研究认识 (P2)

**Goal**: 显式补证关联旧结论，保留原循环预算/原标准/终态，并可协调中断提交。

**Independent Test**: 在临时 10/10 终态 loop 上提交 follow_up Finding/refines，单次主题查询看到局部收益与后续限制；原 checkpoint bytes、轮次、criterion 和终止 assessment 不变；精确重放不会重复创建。

### Tests

- [x] T041 [P] [US4] 在 tests/integration/test_autoresearch_loop.py 添加终态后 follow_up/source/relation 的显式提交与 topic/show/context 查询，比较原 checkpoint 全 bytes/预算/标准/assessment，核对后续历史扩展不冒称未来独立确认。
- [x] T042 [P] [US4] 在 tests/unit/test_research_store.py 添加后续提交已成功但回执丢失、文件后来缺失仍可原 key 重放、并发相同 key、不同 payload 冲突及旧不可核验 receipt 的协调场景。

### Implementation

- [x] T043 [US4] 在 src/aimx/research/validation.py、src/aimx/research/replay.py 和 specs/007-research-state/contracts/research-update.schema.json 完成 activity.create 的 follow_up/related_loop_id 校验，允许来源独立的 Finding/relation 指向旧 Finding；终态补证不调用 control.py 的 begin-round/checkpoint。
- [x] T044 [US4] 在 src/aimx/research/overview.py、src/aimx/commands/finding.py 和 src/aimx/rendering/research_views.py 单列 follow_up 活动/成果导航，topic 查询的相关修正并列原收益与后续范围，明确原 loop 归属/目标事实不变；保留 context 的跨活动闭包。
- [x] T045 [US4] 更新 skills/aimx-autoresearch/references/research-handoff.md，给出新 Activity+Finding+relation 和稳定 update key 的终态后补证步骤，说明 receipt 快照 not_found 的并发限制、精确重放及冲突后刷新重审，不重开或扩轮。
- [x] T046 [US4] 运行 T041/T042 和 quickstart.md 的 follow-up 验收，记录 SC-005、FR-012/FR-013 的查询结果、终态 bytes 比较与提交协调证据于 specs/008-autoresearch-knowledge-loop/validation.md。

**Checkpoint**: US4 无训练执行要求，单次后续产物更新可独立证明终态不可变。

## Phase 7: User Story 5 — 用明确的研究合同约束主动探索 (P2)

**Goal**: 派发前有完整合同、实际能力预检与可解释观察状态；执行、产物、科学评估、审计和资源事实分别交接。

**Independent Test**: 假宿主/evaluator 两轮流程中，不完整合同/缺能力/已观察数据的独立性声明阻止派发，合法探索可继续；至少五类故障保留证据和四轴差异，失败策略提前验证，slot/model/预算不重放或重置。

### Tests

- [x] T047 [P] [US5] 在 tests/contract/test_autoresearch_protocol.py 添加完整/缺项实验合同与预检报告、数据 scope/freeze/first_scored/development 事件、Attempt 四轴及 resource source/enforcement 的合同用例；保留旧 config/checkpoint/result 形状和真实 requested/actual 来源规则。
- [x] T048 [P] [US5] 在 tests/integration/test_autoresearch_skill_loop.py 扩展假宿主/evaluator，覆盖缺命令/模型/预检、无 Aim SDK 的文件研究、已评分/转开发/未知/重叠独立性拒绝及合法 exploratory、并发观察后评分复核、五类故障与继续/停止策略；不连接在线模型或 GPU。

### Implementation

- [x] T049 [US5] 在 src/aimx/research/models.py 和 src/aimx/research/ids.py 定义 DatasetScope/Observation/ExecutionOutcome 投影和身份、Agenda 可选 experiment_contract，明确四轴与资源验证枚举/unknown，保留旧 Agenda 和 checkpoint.result 的既有形状。
- [x] T050 [US5] 在 src/aimx/research/events.py、src/aimx/research/validation.py、src/aimx/research/replay.py 和 specs/007-research-state/contracts/research-update.schema.json 实现 dataset.scope.create/dataset.observation.record 及 strict experiment_contract 校验：来源、覆盖/overlap、冻结候选/attempt、首次评分/转开发、迟到冲突与真实事件时间，不能删除观察历史或自动解析 annotation。
- [x] T051 [US5] 在 src/aimx/research/validation.py 和 tests/unit/test_research_validation.py 实现 independent_confirmation 的一致性 gate，拒绝已观察/开发/未知覆盖/无法证明不重叠/冻结不成立，错误含首次事件/来源和查询入口；explicit exploratory 不被独立性 gate 误阻塞，不声称证明记录外未读标签。
- [x] T052 [US5] 在 src/aimx/research/validation.py、src/aimx/research/replay.py、src/aimx/research/events.py 和 specs/007-research-state/contracts/research-update.schema.json 实现 execution.outcome.record 的 activity+attempt_key 去重、retry_of 追溯、四轴独立、verified 资源需执行器依据、超额/日志缺失保留原事实；同 key 改内容拒绝，不覆盖原 attempt。
- [x] T053 [US5] 在 src/aimx/research/overview.py 和 src/aimx/rendering/research_views.py 聚合 Attempt 四轴、请求/实际设置、资源 planned/actual/source、重复/超额/未核验/未强制及原 checkpoint.result 差异，不把科学 passed 改写成执行 succeeded 或 goal met。
- [x] T054 [US5] 更新 skills/aimx-autoresearch/SKILL.md 和 skills/aimx-autoresearch/references/loop-contract.md，规定实际入口/version/help/read/dry-run、所需 Aim/file 能力、所有轮次模型设置、项目 evaluator 的实际预检与 ready 规则；预检不训练，报告经 lead 显式引用，通过既有 artifact/update 回执关联而非修改 config 指纹。
- [x] T055 [US5] 更新 skills/aimx-autoresearch/SKILL.md 和 skills/aimx-autoresearch/references/research-handoff.md，要求完整冻结合同和观察快照在 reserved/spawn 前验证，项目评分启动前复核并发观察；stop_pending/continue_independent 派发前核对宿主，无有效策略停止后续 pending，不替换模型/重放槽位/退款/扩轮，不承诺取消已运行进程。
- [x] T056 [US5] 在 skills/aimx-autoresearch/assets/ 新建 project-capabilities.example.json、experiment-contract.example.json 和 preflight.example.json，展示 argv/evaluator/preflight/资源来源/失败策略及字段缺项提示；保持 skills/aimx-autoresearch/assets/config.example.json 原字段和 schema，示例与合同/测试同步且不嵌入私有路径。
- [x] T057 [US5] 在 tests/unit/test_autoresearch_control.py 与 tests/integration/test_autoresearch_skill_loop.py 固定原 config fingerprint、finished.result、round/slot 单向转换和终态兼容断言；确认新合同/attempt 经研究记录与既有 artifacts 关联，继续旧循环缺项具体阻塞，无 checkpoint 格式转换或新增版本开关。
- [x] T058 [US5] 运行 T047/T048/T051/T057 与 quickstart.md 的两轮假执行器和至少五类故障矩阵，记录 SC-007/SC-008 的拒绝/允许、四轴/预算/回执和 worker 不写共享状态证据于 specs/008-autoresearch-knowledge-loop/validation.md。

**Checkpoint**: 全部五个故事完成；仍只有项目/宿主执行实验，Aimx 不新增训练 runner、调度器或硬计算预算平台。

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 对齐用户说明，证明全特性兼容、安全与可发现性。依赖所有故事完成。

- [x] T059 [P] 更新 README.md 与 skills/aimx/SKILL.md 的研究入口/成果回执/文件证据/单一 CLI 指引，明确 context budget 从 items-only 收紧为完整 JSON 响应及同预算内容可能减少；保留 ownership、read-only、uv 和 native passthrough 说明，不发布 release 或修改全局安装 skill。
- [x] T060 [P] 在 tests/integration/test_passthrough_behavior.py 与 tests/integration/test_missing_python_aim_package.py 补充 native argv/stdout/stderr/exit/help fidelity 和纯 state/file/help 无 SDK 场景，确保所有新入口只读、安全失败且 unknown root 继续透传。
- [x] T061 [P] 更新 specs/008-autoresearch-knowledge-loop/quickstart.md 为真实可执行入口和产物字段，对齐 specs/008-autoresearch-knowledge-loop/contracts/ 四份合同与 specs/007-research-state/contracts/cli-output.md 的统一预算说明；将 tests/fixtures/research_knowledge_loop/README.md 中 fixture 与各 SC 的对应关系补齐，不新增旧格式维护分支。
- [x] T062 运行 specs/008-autoresearch-knowledge-loop/quickstart.md 的 targeted unit/contract/integration/scale 与只读 data/.aim smoke，复核 SC-001 五类三次查看/两分钟、SC-002 全层帮助、SC-003 四结果、SC-004 字节边界、SC-005 终态、SC-006 文件/只读、SC-007 独立性、SC-008 五类故障，将通过/失败/未运行及全部 FR 映射写入 specs/008-autoresearch-knowledge-loop/validation.md；失败不得标完成。
- [ ] T063 对 AGENTS.md 要求的每次符号 impact 记录、最终 GitNexus detect_changes（若准备提交则必须先执行）、git diff --check 与实际文件差异作范围复核，记录 specs/008-autoresearch-knowledge-loop/validation.md，确认未触碰用户原改动、Aim 数据、私有研究状态或双版本/迁移表面；本任务不自动提交或推送。

## Dependencies & Execution Order

### Phase dependencies

```text
Setup T001–T004
        ↓
Foundation T005–T008
        ↓
US1 T009–T020 (P1 MVP)
        ↓
US2 T021–T031 (P1)
        ├───────────────────────────┐
        ↓                           ↓
US3 T032–T040 (P1)          US5 T047–T058 (P2)
        ↓                           │
US4 T041–T046 (P2)                   │
        └──────────────┬────────────┘
                       ↓
               Polish T059–T063
```

US5 的最低功能依赖为 US1+US2；它不依赖 US4 的终态补证。上图按实际功能依赖展示，US5 可在 US2 后设计/准备测试。US4 完整验收依赖 US3 的真实 context；US3 仅需 US2 的证据/来源能力，跨活动闭包用静态 fixture，不依赖 US4 的生产入口。所有故事能在临时数据上独立验证其价值，不要求运行其它故事的真实研究任务。

### Within-phase task prerequisites

- T001 → T002 → {T003,T004} → T005 → T006 → T007 → T008。
- US1：{T009,T010} 为测试 ready wave；T011 → T012 → T013 → T014 → T015 → T016 → T017 → T018 → T019 → T020。实施先确认新增测试按预期失败；业务入口在依赖实现后开放。
- US2：{T021,T022,T023} 为测试 ready wave；T024 → T025 → T026 → T027 → T028 → T029 → T030 → T031。
- US3：{T032,T033} 为测试 ready wave；T034 → T035 → T036 → T037 → T038 → T039 → T040。
- US4：{T041,T042} 为测试 ready wave；T043 → T044 → T045 → T046。
- US5：{T047,T048} 为测试 ready wave；T049 → T050 → T051 → T052 → T053 → T054 → T055 → T056 → T057 → T058。
- Final：全部故事通过后 {T059,T060,T061}，然后 T062 → T063。

未标 `[P]` 的实现任务按上述顺序运行。不同故事反复编辑 models/validation/replay/research_views/skill 等文件，禁止因故事功能独立就并发写同一文件；共享源码一个 writer 或隔离工作树串行集成。即使并行测试任务可独立编写，也要先消费已完成 fixture/合同；标记不代表 tests 必须在功能实现前通过。

## Parallel Examples Per User Story

以下均在各故事依赖满足、共享 fixture 已就绪后启动，只列不同文件的测试编写；任务完成后串行进入实施。它们是可选择的执行划分，不要求当前会话启动 worker。

| Story | 可以同时执行的任务 | 文件隔离 |
| --- | --- | --- |
| US1 | T009 + T010 | tests/contract/test_research_contract.py / tests/integration/test_research_commands.py |
| US2 | T021 + T022 + T023 | tests/unit/test_research_evidence.py / tests/unit/test_research_validation.py / tests/integration/test_research_commands.py |
| US3 | T032 + T033 | tests/unit/test_research_context.py / tests/contract/test_research_contract.py |
| US4 | T041 + T042 | tests/integration/test_autoresearch_loop.py / tests/unit/test_research_store.py |
| US5 | T047 + T048 | tests/contract/test_autoresearch_protocol.py / tests/integration/test_autoresearch_skill_loop.py |

额外机会：Setup T003/T004、Final T059/T060/T061 分别为不同文件的 ready wave。跨故事相同文件测试任务仍不得重叠。例如 T010 与 T023 共享 integration 文件，T009 与 T033 共享 contract 文件，只在不同阶段执行。

## Requirements / Success Criteria Traceability

| Requirement | Tasks | 验收 |
| --- | --- | --- |
| FR-001 / FR-002 | T010–T020、T053–T055 | SC-001 / SC-008，三种状态分离、所有停止路径与部分回执 |
| FR-003 | T009、T015–T016、T020、T060 | SC-002，全层 help 无 IO |
| FR-004 | T010、T017–T018 | SC-001 / SC-006，非 Finding/合法空/坏位置 |
| FR-005 / FR-006 / FR-007 | T021–T027、T029–T031 | SC-003 / SC-006，四类语义与三个证据轴 |
| FR-008 | T023、T027–T030、T044 | SC-003 / SC-005，主题/归属/提炼/相关修正 |
| FR-009 / FR-010 / FR-011 | T032–T040 | SC-004，完整闭包/全响应预算/历史文本限制 |
| FR-012 | T041、T043–T046 | SC-005，终态不变、后续认识可见 |
| FR-013 | T006、T013、T019、T042、T045 | SC-008，精确重放/冲突/原子/不可核验回执 |
| FR-014 / FR-015 | T047–T051、T054、T056、T058 | SC-008，合同/实际入口/宿主/evaluator 预检 |
| FR-016 | T048、T050–T051、T055 | SC-007，观察/冻结/评分/转开发/未知与重叠 |
| FR-017 / FR-018 | T047–T049、T052–T053、T057–T058 | SC-008，四轴、多 attempt/资源能力诚实性 |
| FR-019 | T048、T054–T055、T057–T058 | 预声明失败策略、宿主规则和 no-replay |
| FR-020 / FR-021 | T004–T010、T020、T031、T060–T063 | SC-006，旧输入/历史、人类/JSON、只读和透传 |

## Implementation Strategy

**MVP first**: 完成 T001–T020（Setup/Foundation/US1），在五类静态样例上验收并演示成果可发现性。MVP 不宣称支持文件 Finding 或新实验协议，尚不满足全特性完成标准。

**Incremental delivery**: US2 增加真实文件证据知识 → US3 完成可信上下文，形成全部 P1 → US4 补证与 US5 协议按上述依赖交付 → Final 验收全部 FR/SC。P2 属于本 spec，不能因交付 MVP 就标全功能完成。

**Execution evidence**: 新增测试先证明失败点正确，实施后运行针对性命令；仅在变更/失败/未解疑问时扩大验证。所有勾选必须有实际文件和验证证据，环境失败不标任务完成。GitNexus impact/detect_changes 为实施和提交约束，本次文档生成未修改程序符号。

## Generation Validation & Extension Hooks

任务总数 63：Setup 4、Foundation 4、US1 12、US2 11、US3 9、US4 6、US5 12、Final 5。16 个 `[P]` 任务分属 7 个 ready wave；所有任务均含顺序 ID、明确文件路径，故事阶段均含正确 `[USn]` 标签，非故事阶段无故事标签。

before_tasks 和生成后的 after_tasks 均已检查 `.specify/extensions.yml`，文件不存在，按规则跳过；没有待执行必需钩子，不创建配置或自行执行未注册命令。编号/格式/故事标签/路径、相对链接、FR/SC 覆盖和无双版本残留检查已通过，git diff --check 通过。本阶段只生成任务文件并校验，不执行以上实施任务。
