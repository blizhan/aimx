# Research Context：统一选择与预算合同

```text
aimx research context --objective TEXT --budget BYTES [--repo PATH] [--loop LOOP_ID] [--json]
```

只有一套选择和预算行为，用户无需选择协议版本。`--budget` 统一约束完整 JSON 响应，`--loop` 为可选来源筛选，skill 沿用原命令调用方式。schema_version 保持 1；compiler_version 更新为 context-v2，仅标识选择算法，不提供旧编译器切换。读取不改变已存 Finding/annotation。

兼容影响：现有实现/007 合同/单元测试按 items-only 计预算，仓库 skill 直接使用返回上下文，未发现按旧计数拆包的产品代码。因此本期明确收紧为完整响应预算，同一预算可能容纳更少内容或报不足；更新相关文档和字节边界测试，不维护旧行为分支。无法核查仓库外调用方，release/README 必须说明这项语义变化。

## Selection

1. 读取同一 revision 的知识快照，不进行在线模型推理、哈希全量文件或写入。
2. 归一化 objective/topics/claim/scope 文本；Unicode casefold，英文词项匹配，中文采用长度至少 2 的连续片段匹配。必须有文本命中才能形成 seed，状态加权只能在匹配项内排序。固定算法以 compiler_version 标识。
3. 在匹配 seed 中按词法得分降序、最新影响解释的 revision 降序、stable ID 升序排名。指定 loop 仅筛 seed 的 source_activity_id；无归属种子只在没有 loop 过滤或显式 annotations 入口出现，不默认为当前 loop。
4. 对 seed 求 active challenges/refines/supersedes 的递归无向闭包，包含每个相关 Finding 的 claim/result_kind/scope/limitations/epistemic/confidence、必要 relation/审核/与实体明确关联的 annotation；visited 防环。跨活动修正标 `closure_dependency:true`，不能被 loop filter 删除。
5. 最相关完整知识组优先。每个候选组全部适配才选择；不得选 seed 后省掉修正。最相关必需组不能容纳则 budget error。其它被完整省略的组在 omission 计数与原因中说明。supports/tests/related_to、Frontier/Agenda 可在必要组后完整加入，不能单独提供有未知/遗漏重要关系的 Finding。
6. annotation-only 历史用词法相关性和较新 revision 确定性排序；明确 target 同实体的记录作为解释组。仅有自由文本时选相关最新结果及相关旧记录，并输出 `legacy_relationships_unverified` 限制；不能自动声明它们是支持/否定关系。预算至少保留限制/遗漏说明，不能谎称修正已完整闭合。
7. evidence_refs 覆盖所有已选结构化引用，按 identity 去重保留用途映射和来源；普通文本路径不加入 refs。当前可用性能廉价确认时返回，无法确认为 unknown/not_checked；科学 review 来自持久历史而非文件存在。选择本身不受可用性波动影响，相同状态/查询/证据状态输入序列化一致。

重组或摘要只能剔除重复历史展示，不能缩写关键 claim/范围/限制/关系以使不完整知识看似完整；完整历史由明确 query_commands 获取。

## Envelope

```json
{
  "schema_version":1,
  "compiler_version":"context-v2",
  "repo":"/study",
  "revision":16,
  "objective":"候选收益",
  "loop_id":null,
  "budget_unit":"utf8_bytes",
  "budget_scope":"response",
  "budget_limit":18000,
  "budget_used":0,
  "items":[],
  "evidence_refs":[],
  "selection_notes":{
    "policy":"lexical_relevance_then_material_closure",
    "selected_groups":[],
    "omitted_count":0,
    "omitted_by_reason":{},
    "limitations":[],
    "query_commands":[]
  }
}
```

上例 budget_used 是待计算示意值；实际必为精确字节数。selected_groups 含 seed IDs、完整闭包 IDs、理由；大量 omissions 只计数和原因，全部记录由 query 入口获取。匹配为空时明确 `no_matching_knowledge`，可显示相关政策入口但不伪造匹配 Finding。

## Budget / deterministic serialization

JSON 输出使用 `ensure_ascii=False,sort_keys=True,separators=(",",":")` 的紧凑格式 + 一个 LF。测量包括整个实际 stdout 响应：所有 items、metadata、refs、selection_notes、数字和 LF。将 budget_used 迭代到固定点再测；`len(stdout.encode('utf-8')) == budget_used <= budget_limit`。排序不使用读取时刻；临时 checked_at 不出现在 context，独立 evidence check 可以报告检查时刻。

Human 输出使用同一选择结果，内部 JSON 选择预算 scope=response_json 明确显示，终端装饰不提供给机器预算消费者；机器交接只能用 --json。不得把 Human 字符数称为 UTF-8 预算。

最小 envelope 或最相关必要组无法容纳时 exit 2，stdout 为空；stderr 既有 error envelope，code 为 budget_too_small_for_required_context，details 含 `budget_limit/required_bytes/required_group_ids/query_commands`。stderr 诊断不属于成功上下文预算且不携带部分结论，tiny budget 也能读到完整不足原因。不可把 budget 失败解释为不存在研究成果。

## Acceptance boundary

必须覆盖修正链 A→B→C、环、跨 loop 修正、18,000-byte 的十轮 fixture、N/N-1 边界、中文 query、全未匹配但 accepted 的 Finding、纯 annotation 最新修正、大量 omissions 和混合证据类型。原命令调用方式保持有效；旧 items-only 字节断言改为完整响应的统一断言，不维护双轨测试。大型闭包不可随意 depth limit；超预算明确失败。
