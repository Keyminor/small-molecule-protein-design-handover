# RFdiffusion整合NISE算法

以下代码从《RFdiffusion整合NISE算法》技术文档移出。


## 步骤1：准备两个环境

```bash
# 在 NISE 仓库根目录
bash setup.sh
# 解释器：./.venv/bin/python
```

```bash
# Foundry 的 rfd3 另装，例如服务器上的
# /root/autodl-tmp/foundry-venv/bin/rfd3
```

```bash
# 服务器上的工作目录
# 代码  /root/autodl-tmp/NISE
# Boltz 缓存  /root/autodl-tmp/.cache/boltz
# 显卡只跑一个任务  CUDA_VISIBLE_DEVICES=0
```


## 步骤2：把药物写成 RFdiffusion 能读的配体

```yaml
oxaliplatin_oxalate_150_200:
  input: ../ligand/conformers/rfd3_inputs/oxaliplatin_oxalate_face.pdb
  ligand: l:g
  contig: 150-200,/0,C1-1
  redesign_motif_sidechains: false
  infer_ori_strategy: hotspots
  is_non_loopy: true
  select_hotspots:
    C1: C1,C2,O1,O2,O3,O4
  select_buried:
    C1: C1,C2,O1,O2,O3,O4
  select_exposed:
    C1: C5,C6,C7,C8
  select_hbond_acceptor:
    C1: O3,O4
```


## 步骤3：用 RFdiffusion3 生成骨架

```bash
cd /root/autodl-tmp/NISE
.venv/bin/python D:\工作汇报\rfd3_nise/run_rfdiffusion3.py \
  --rfd3-executable /root/autodl-tmp/foundry-venv/bin/rfd3 \
  --inputs-yaml D:\工作汇报\rfd3_nise/job_recommended.yaml \
  --output-dir campaigns/oxaliplatin/rfd3_outputs_pocket \
  --scale pocket \
  --diffusion-batch-size 4 \
  --dry-run
```

```bash
# 确认无误后去掉 --dry-run。不要同时再开第二个 GPU 任务。
```


## 步骤4：把骨架转成 NISE 复合物

```bash
.venv/bin/python D:\工作汇报\rfd3_nise/prepare_rfd3_outputs.py
```

```bash
# 单条骨架也可以单独转：
.venv/bin/python D:\工作汇报\rfd3_nise/prepare_rfdiff_pose_for_nise.py \
  --input-pdb path/to/rfd3_pose.pdb \
  --output-pdb path/to/prepared_pose.pdb
```


## 步骤5：用 Boltz 预筛，只留一条进入 NISE

```bash
.venv/bin/python D:\工作汇报\rfd3_nise/rank_rfdiff_poses_with_boltz.py \
  --prepared-dir campaigns/oxaliplatin/pocket_run/prepared \
  --output-dir campaigns/oxaliplatin/pocket_run/boltz_prefilter \
  --nise-input-dir campaigns/oxaliplatin/nise_run_pocket \
  --smiles 'O=C1O[Pt]2(OC1=O)N[C@@H]1CCCC[C@H]1N2' \
  --boltz-executable D:\工作汇报\rfd3_nise/boltz_allow_platinum.py \
  --boltz-devices cuda:0 \
  --boltz-cache /root/autodl-tmp/.cache/boltz \
  --sampling-steps 50 \
  --chunk-size 16 \
  --cleanup-predictions
```


## 步骤6：运行 NISE

```bash
.venv/bin/python D:\工作汇报\rfd3_nise/run_nise_boltz2x_oxaliplatin.py \
  --input-dir campaigns/oxaliplatin/nise_run_pocket \
  --boltz-devices cuda:0 \
  --num-iterations 35
```

```bash
# 只打印参数、不启动循环：
.venv/bin/python D:\工作汇报\rfd3_nise/run_nise_boltz2x_oxaliplatin.py \
  --input-dir campaigns/oxaliplatin/nise_run_pocket \
  --dry-run
```


## 步骤7：在已过门的复合物上做部分扩散

```bash
bash D:\工作汇报\rfd3_nise/server_launch_partial_diffusion.sh
```

```bash
bash D:\工作汇报\rfd3_nise/server_run_rescore.sh boltz
```
