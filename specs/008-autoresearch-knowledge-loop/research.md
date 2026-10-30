# Phase 0 Research: Autoresearch 研究成果沉淀与可靠交接

本文件区分仓库现状与本期决定。研究依据为实际源码、既有合同和测试；两名 `magpie/codex/gpt-6-luna` worker 仅收集信息，父会话负责决策。无领域实验、历史数据提炼或研究状态写入。

## 1. 治理、路径和实现基线

**Decision**: 在实际分支 `main` 规划目录 `specs/008-autoresearch-knowledge-loop/`。setup 返回的同名 `BRANCH` 是功能标识，不声明已经创建 Git 分支。采用根目录 `CONSTITUTION.md` v1.0.0 的已填充原则检查设计。

**Rationale**: spec 明确目录与分支独立；`.specify/memory/constitution.md` 仍为原始占位模板，不能用未批准的占位内容替换已有安全规则。两份文件均已读取，差异在 plan 留痕，不扩大本期范围去修改治理文件。

**Alternatives considered**: 自动创建分支、覆盖 memory constitution、沿用旧 007 plan 中的错误分支/机器路径，均不符合当前 spec 和最小改动原则。

**Evidence**: `spec.md` 的 Feature Branch/Assumptions；根 Constitution；`.specify/scripts/bash/common.sh::get_feature_paths`；实际 `git branch --show-current` 为 `main`。技术基线来自 `pyproject.toml`：Python `>=3.10,<3.13`，开发使用 3.12；pytest、Aim SDK 属于 dev group。无新运行时依赖。

## 2. ResearchUpdate、事件回放与兼容

**Decision**: 用户已确认采用单一 CLI 与增量数据扩展。保留 schema_version=1 和现有 envelope，既有 operation 新增可选字段，新增实体以明确新 operation 表达；直接扩展同一解析/回放路径。旧载荷原样有效，提供新字段时严格校验，008 工作流在生成成果和派发实验时检查完整性。SQLite 沿用 commits/events/metadata，不迁移旧事件、不新增知识数据库。compiler_version 仅标识算法，不提供切换。

**Rationale**: 当前模型/事件 payload 支持增量字段，真正的兼容需求是新程序读旧数据、旧调用尽量有效，无需两套用户协议。既有缺字段的记录在内存显示 unknown/unattributed，不能静默重新解释。新文件证据和新增事件的必需字段可以按 kind/op 校验，不需要更换 envelope。

**Alternatives considered**: 双版本 update/context、版本协商和 checkpoint 格式升级增加调用方、校验、文档及测试分支，用户已决定移除；全部藏进 annotation 无法检索；另建数据库增加双写。旧程序理解未来事件不在兼容承诺内，未知事件必须安全报错。

**Evidence**: `src/aimx/research/models.py::ResearchUpdate/ResearchState`、`validation.py::compile_update/_validate_update_header`、`events.py::SUPPORTED_OPERATIONS`、`replay.py::replay/apply_event`、`store.py::_SCHEMA`；`specs/007-research-state/contracts/research-update.schema.json`。

## 3. 活动归属、概览和回执

**Decision**: 新记录显式引用 `source_activity_id`。Activity 为不可变来源登记，类型为 `loop`、`analysis` 或 `follow_up`，记录产物/报告入口；loop 类型引用标准 checkpoint，follow_up 显式引用原 loop。checkpoint 是轮次、槽位、原验收与终止状态的事实源；事件存储是知识与提交事实源。overview 只读聚合二者，不持久化一份独立 Receipt 状态。

旧记录无活动标识时属于 `unattributed`，不按时间、文本中的 loop 名或最近一次循环猜测。指定 loop 只返回其来源记录；有关联的后续活动在单独 follow-up 区域展示。丢失 checkpoint、状态不可读与合法空数据库是不同结果。

**Rationale**: 当前 loop 身份只在 `control.py` checkpoint 内；Agenda `constraints.loop_id` 是任意 JSON，不能覆盖整个知识图。Receipt 必须允许“产物已保存、提交不确定”，不能把 round 完成当成目标达成。

**Alternatives considered**: 把所有 repo 记录归最新 loop 会串数据；把 receipt 写成第二份事实会漂移；overview 自动扫描并导入外部目录违反显式写入。

**Evidence**: `skills/aimx-autoresearch/scripts/control.py::init/load_state/begin_round/checkpoint`；`references/research-handoff.md`；`rendering/research_views.py::render_state_human/render_findings`。现有空 Finding 只显示 `No findings.`，需要非 Finding 计数和查询入口。

## 4. 文件证据和科学语义

**Decision**: 扩展 EvidenceReference 为 `aim_run | local_artifact` 联合。文件引用保留原位置、规范化绝对位置、SHA-256、字节大小、类型、用途和来源活动。相对路径只按显式 repo 根解析；允许 repo 外本地文件，不搬运文件或改 Aim。登记时流式读取普通文件并校验调用者声明的 digest/size；记录 symlink 解析后的身份路径，拒绝目录和特殊文件，读取期间文件变化则拒绝本次写入。

可用性 `available/missing/unreadable/unsupported/unknown`、身份 `matches/changed/not_checked/unknown`、科学审核 `not_reviewed/accepted/rejected/inconclusive` 是不同轴。普通读取只返回已知状态或 not_checked，不自动全量哈希；显式 evidence check 重新读取但不写状态。科学审核需显式追加 review 事件，哈希一致绝不自动升级置信度。

Finding 增加结果性质 `improvement/degradation/insufficient_evidence/scope_boundary`、研究范围、限制、topics 和来源；保留既有 epistemic/governance/confidence。区间跨零可用 insufficient_evidence + candidate，不强制 contradicted。历史 annotation 显式提炼使用新 Finding 的 `source_annotation_ids`，校验原记录存在；不提供自动转换。

**Rationale**: 当前 `_canonical_evidence` 只接受 Aim run，`mark_evidence_availability` 把所有 Aim 访问失败压成 unavailable。SHA-256 表达内容身份，不表达统计有效性；新文件引用不需安装 Aim SDK。

**Alternatives considered**: 伪造 Aim run、只存自由路径、只存 mtime、自动修正变化的 digest 均不满足追溯；自动内容复制/远程抓取超范围。

**Evidence**: `src/aimx/aim_bridge/research_evidence.py`；`validation.py::_canonical_evidence`；`models.py::EvidenceReference/Finding`；`tests/unit/test_research_evidence.py`。哈希采用标准库 hashlib，无新依赖。

## 5. 相关性、修正闭包和字节预算

**Decision**: 统一 context 行为，内部 compiler_version 更新为 `context-v2`，使用确定性词法选择（Unicode 归一化、英文词项及中文连续片段）；必须先命中问题/topics/scope 才能成为 seed，治理状态只在匹配项内作排序加权。对 `challenges/refines/supersedes` 递归求无向连通闭包，用 visited 去重并防环，包含关系、必要范围和审核；supports/tests/related_to 只作可选上下文。loop 过滤限制 seeds，不截断跨活动已知反证；闭包外来项标记 closure dependency。

最相关的必要知识组先打包，其后按固定得分、最新修正 revision、ID 顺序选择完整组。任何选中 seed 的必要闭包都必须完整；首个必要组或返回必需元数据无法容纳时返回明确 budget error。未入选完整组计入 omissions，不返回它的单边 seed。历史 annotation 单独按相关性、目标关联和后续 revision 排序；与同一明确实体相关的记录作为解释组，无法证明关系的自由文本只标限制，不伪造图边。

`--budget` 统一计整个实际 JSON 响应的 UTF-8 字节数（含 envelope、evidence_refs、selection_notes 和换行）；紧凑稳定序列化，budget_used 迭代到固定点后测量。命令调用方式不变，不提供格式选择或旧行为回退。失败诊断在 stderr，明确不属于成功上下文预算，永不夹带部分知识。

调用方核对：`skills/aimx/references/autoresearch-protocol.md` 和 007 合同明确旧 items-only 语义，`tests/unit/test_research_context.py` 与 `test_research_scale.py` 直接断言 budget_used=json_bytes(items)，需要随实施更新；仓库 skill 与 `test_autoresearch_skill_loop.py` 使用返回 items/revision，未发现依赖旧计数拆包的产品代码。无法证明仓库外没有依赖，因此 README/release 明确说明：同一 budget 在新规则下可能减少内容或返回不足；提供更严格、单一预算保证，不维护双轨。

**Rationale**: 现有 `_closure_ids` 仅一层，`_finding_score` 未命中也可能因状态加权成为 seed；`_policy_items` 以顺序塞入全局 annotation；`json_bytes(items)` 不计其它字段而 renderer 又输出缩进 JSON。固定快照与查询必须重复产生同一结果，选择不依赖当前时钟、文件检查时刻或在线模型。

**Alternatives considered**: 只按最新 revision 会丢相关基础；截断文本或闭包会改变认识；embedding/LLM 选择增加依赖和不确定性；保留旧预算和新预算两套路径扩大维护成本；经调用方核对后采用明确记录的统一语义变化，而非增加版本开关。

**Evidence**: `src/aimx/research/context.py::compile_context/_finding_score/_closure_ids/_policy_items/_evidence_refs`；`rendering/research_views.py::render_json`；`tests/unit/test_research_context.py`、`test_research_scale.py`。现有 scale fixture 为 10,000 commits / 50,000 events。

## 6. 提交协调和终止后补证

**Decision**: 复用 `client_update_id`、原请求指纹、原回执和 `BEGIN IMMEDIATE`。先保存确切 update 文件及 SHA-256，再提交，再记 checkpoint 回执；不宣称 SQLite/JSON/产物三者原子提交。新增只读 receipt 查询按 key 取回已提交结果并校验 fingerprint；不确定时也可继续使用现有精确重放路径。不同 payload 不能复用 key；真正 revision conflict 必须刷新和重新审议后换 key，不能只改 base_revision。

终止后新知识使用独立 follow_up Activity 与新的提交 key，新增 Finding/关系指向原 Finding，不更新原 loop checkpoint。原 snapshot 与原目标结果保留；overview 并列显示后续认识。

**Rationale**: `apply_update` 在 revision/evidence 重新验证之前检查已提交 key，已能恢复成功但回执丢失的提交，文件后来丢失也不能否认历史提交。`_idempotent_result` 不能证明内容指纹的旧回执须报无法协调，不猜测成功。

**Alternatives considered**: 自动重试 stale base、重开第 11 轮、更新终态 snapshot、重新哈希后替换旧请求都会改写原事实。

**Evidence**: `src/aimx/research/store.py::apply_update/_existing_client_commit/_request_fingerprint/_idempotent_result`；`skills/aimx-autoresearch/references/research-handoff.md`；现有 autoresearch integration 幂等测试。

## 7. 实验合同、数据观察和 attempt

**Decision**: AgendaItem 保留原松散字段，可选增加严格 `experiment_contract`：问题、干预、固定对照、预期证据、反证条件、混杂、评估协议身份与预检、数据范围/阶段、候选冻结和计算预算声明。登记 DatasetScope 和 Observation 事件（声明、冻结、首次评分、转开发），查询用于独立确认检查。scope 的 overlap key/范围由项目提供；重叠范围任一评分/开发事件都污染独立性，无覆盖记录或无法判断重叠时为 unknown，拒绝独立确认声明，但允许明确 exploratory。

执行器不在 Aimx 内运行。不可重放 slot 可有执行器报告的多个 Attempt，各自有 ID、原 attempt 来源、执行状态、产物完整性、评估状态、审计状态、模型实际设置和资源计数。重试日志必须暴露；助手不得因日志允许多 attempt 而自动重放宿主 slot。Checkpoint 保存派发事实/attempt 引用，结构化科学结果追加到 Research State；聚合 receipt 暴露不一致，不覆盖原结果。

**Rationale**: 现有 Agenda 足以复用任务入口但缺严格合同；`validate_result` 单个 status 无法表达服务失败但预测完整；当前记录不含数据观察身份。记录外标签读取无法被 Aimx 证明，报告中必须保留这项限制。

**Alternatives considered**: 新实验调度平台超范围；用一次 status 或 Agenda completed 判断科学成功错误；仅按 split 名/当前日期判断独立性不可靠。

**Evidence**: `models.py::AgendaItem`、`validation.py`；`control.py::validate_result/patch_agent`；`skills/aimx-autoresearch/references/loop-contract.md`；`tests/contract/test_autoresearch_protocol.py`。

## 8. 预检、失败策略与旧循环接入

**Decision**: skill 在保留 config shape validate 的基础上做实际能力预检，逐项记录使用的 Aimx 绝对入口/包版本与实际命令能力、必要 read/update/dry-run/help 能力、Aim 能力（仅需要 Aim evidence 时必需）、每轮模型/effort 支持、项目 evaluator 预检报告与宿主规则。只检查所需能力，缺项返回 missing/unverified，不静默降级或替换模型。预检执行项目明确提供的检查命令，不派发实验、不修改源码/生产配置。

实验合同/preflight 明确 `failure_policy=stop_pending|continue_independent`；后者必须有宿主支持证据。缺声明/未知宿主策略停止后续派发，既有正在运行任务按照宿主处理，不承诺自动取消。保留 begin-round 先消费预算和 slot pending→reserved→started→finished 单向状态。

现有 config/checkpoint/schema_version/fingerprint/result 形状保持，不增加 upgrade 流程。新 preflight/合同/Attempt 四轴/update digest 通过研究记录与既有 artifacts/update_file/research_update 回执关联；继续旧循环前核对关联记录，缺项给具体原因，不能改原配置、已完成 slot 或终态。静态版本能力命令也移除，复用 version/help/read/dry-run 做实际能力检查。

**Rationale**: 现有 skill 规定“一项失败不取消其它任务”，而 FR-019 要求派发前明确已验证策略；本期更新这个默认，不临时调整失败后的模型/预算。轮次硬限制由 control 实现，GPU/token/计算限制只有执行器证据足够时才标 enforced/verified。

**Alternatives considered**: 把 JSON validate 当环境就绪、强制安装 Aim 来读 sidecar、失败后临时退款/重放/替换模型均不满足合同。

**Evidence**: `skills/aimx-autoresearch/SKILL.md` Prepare/Execute/Resume；`control.py::normalize_config/validate/begin_round/checkpoint`；`README.md` Bounded multi-agent research；`tests/integration/test_autoresearch_skill_loop.py` 的假执行器。

## 9. 帮助、合同与验证

**Decision**: 在所有已有研究 command handler 的 repo/ID/SDK 访问前解析该层级 `-h/--help`。只在 Aimx-owned research/finding/lineage/frontier 树内生效，不改变 native Aim 帮助透传。提供静态用途、读写性质、参数、示例、退出码；新接口 human/JSON 双输出，期望错误沿用 0/1/2/3 退出码。

以可移植临时目录和模拟 evaluator/host 验收，覆盖 spec 全部 FR/SC。planning 只校验文档链接、合同例子和覆盖关系；不声称尚未实现的行为已通过测试。后续实施需 GitNexus impact-before-edit 和 detect_changes-before-commit。

**Rationale**: `run_finding_command` 在识别具体 action 之前读 repo；现有顶层 help 未覆盖每层级语义。私有 ICL 报告只作为问题来源，不成为验收依赖。

**Alternatives considered**: 全局拦截所有 `--help` 会破坏透传；实际重跑研究既昂贵又超出本轮设计。

**Evidence**: `commands/{research,finding,lineage,frontier,help}.py`、`router.py`；`tests/integration/test_passthrough_behavior.py`、`tests/contract/test_research_contract.py`。

## 研究收敛

Technical Context 的五组未知项均已获得明确决定：增量兼容扩展、文件身份/检查边界、统一预算与完整关联、提交协调和补证、项目能力/数据观察合同。未留下阻止 Phase 1 的待澄清问题。具体项目 evaluator、统计阈值和宿主模型属于未来配置输入，缺失时按合同阻塞派发，不是规划未决项。
