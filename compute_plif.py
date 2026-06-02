"""
改进版 PLIF（6维相互作用指纹）：增加几何约束

改进点 vs 旧版：
  1. 氢键：增加 D-H···A 角度 ≥ 120° + H···A ≤ 2.5Å
  2. π-π堆积：增加环质心距离 + 环平面夹角检查
  3. 金属配位：配位原子必须是 N, O, S
  4. 疏水/盐桥：保持原有距离规则（无强方向性约束）

统一用 RDKit 处理蛋白和配体（含加氢），不再依赖 Biopython。
"""

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)


# ===================== 辅助函数 =====================

def _compute_plane_normal(points):
    """用 SVD 计算点集的最佳拟合平面法向量（单位向量）"""
    pts = np.array(points)
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    return vh[2]  # 最小奇异值对应的右奇异向量


def _angle_between(v1, v2):
    """两向量夹角（度），范围 [0, 180]"""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# 蛋白芳香残基的环原子名（用于 π-π 检测）
_AROM_RING_DEFS = {
    'PHE': {'6ring': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']},
    'TYR': {'6ring': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']},
    'HIS': {'5ring': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']},
    'TRP': {
        '6ring': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        '5ring': ['CG', 'CD1', 'CD2', 'NE1', 'CE2'],
    },
}

# 蛋白 H-bond 供体原子名（有极性氢的重原子）
HBOND_DONOR_ATOMS = {
    'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ',
    'OG', 'OG1', 'OH', 'SG',
}

# 蛋白 H-bond 受体重原子名（有孤对电子的 O/N/S）
HBOND_ACCEPTOR_ATOMS = {
    'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
    'ND1', 'NE2', 'SD', 'SG',
}

# 氨基酸分类
HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO', 'TYR', 'CYS'}
POSITIVE_RES    = {'LYS', 'ARG', 'HIS'}
NEGATIVE_RES    = {'ASP', 'GLU'}
AROMATIC_RES    = {'PHE', 'TYR', 'TRP', 'HIS'}
METAL_ELEMENTS  = {'ZN', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU'}

# 截断半径
CUTOFF_HYDROPHOBIC = 4.5
CUTOFF_HBOND       = 3.5   # D···A 距离
CUTOFF_HBOND_HA    = 2.5   # H···A 距离
# D-H···A ≥ 120° ⟺ D→H 与 H→A 夹角 ≤ 60°
CUTOFF_HBOND_DEV = 60.0   # 偏离线性的最大角度
CUTOFF_PI_CENTROID = 5.5   # 环质心距离
CUTOFF_SALT        = 4.0
CUTOFF_METAL       = 3.0


# ===================== 主函数 =====================

def compute_simple_plif(protein_pdb_path, ligand_path):
    """
    计算改进版蛋白-配体相互作用指纹（6维向量）

    参数:
        protein_pdb_path: 蛋白 PDB 文件路径
        ligand_path:      配体 .mol2 或 .sdf 文件路径

    返回:
        6维 numpy 数组 [疏水, H供体, H受体, π-π, 盐桥, 金属]
        失败返回 None
    """
    # ========== 1. 加载配体（保留氢） ==========
    try:
        if ligand_path.endswith('.mol2'):
            mol_lig = Chem.MolFromMol2File(ligand_path, removeHs=False)
        else:
            mol_lig = Chem.MolFromMolFile(ligand_path, removeHs=False)
    except Exception:
        return None

    if mol_lig is None:
        return None

    mol_lig = Chem.AddHs(mol_lig, addCoords=True)
    conf_lig = mol_lig.GetConformer()

    # 分离配体的重原子和供体氢
    lig_heavy_atoms = []       # RDKit Atom 对象
    lig_heavy_pos = []         # 对应坐标 (Nx3)
    lig_donor_pairs = []       # [(donor_atom_idx, H_position), ...] 供体重原子+其H

    for atom in mol_lig.GetAtoms():
        idx = atom.GetIdx()
        pos = np.array(conf_lig.GetAtomPosition(idx), dtype=float)
        if atom.GetSymbol() == 'H':
            # 找出此 H 所属的供体重原子（O/N/S）
            for nb in atom.GetNeighbors():
                if nb.GetSymbol() in ('O', 'N', 'S'):
                    lig_donor_pairs.append((nb.GetIdx(), pos))
                    break
        else:
            lig_heavy_atoms.append(atom)
            lig_heavy_pos.append(pos)

    if len(lig_heavy_pos) == 0:
        return None

    lig_heavy_pos = np.array(lig_heavy_pos)
    n_lig_heavy = len(lig_heavy_pos)

    # 配体芳香环信息（用于 π-π）
    lig_rings = []
    ri_lig = mol_lig.GetRingInfo()
    for ring_indices in ri_lig.AtomRings():
        is_arom = all(mol_lig.GetAtomWithIdx(i).GetIsAromatic() for i in ring_indices)
        if is_arom:
            ring_pts = [np.array(conf_lig.GetAtomPosition(i), dtype=float)
                        for i in ring_indices]
            lig_rings.append({
                'centroid': np.mean(ring_pts, axis=0),
                'normal': _compute_plane_normal(ring_pts),
            })

    # ========== 2. 加载蛋白（RDKit + 加氢） ==========
    try:
        mol_prot = Chem.MolFromPDBFile(protein_pdb_path, removeHs=False,
                                       sanitize=False)
    except Exception:
        return None

    if mol_prot is None:
        return None

    mol_prot = Chem.AddHs(mol_prot, addCoords=True)
    conf_prot = mol_prot.GetConformer()

    prot_heavy_pos = []
    prot_info = []             # [{residue, atom_name, element}, ...]
    prot_donor_pairs = []      # [(donor_atom_idx, H_position), ...]
    prot_acceptor_indices = [] # 受体原子在 prot_heavy_pos 中的索引
    # 按残基收集芳香环原子坐标
    arom_res_atoms = {}  # (chain, resnum, resname) -> {ring_label: [positions]}

    for atom in mol_prot.GetAtoms():
        idx = atom.GetIdx()
        pos = np.array(conf_prot.GetAtomPosition(idx), dtype=float)
        elem = atom.GetSymbol()

        mi = atom.GetMonomerInfo()
        if mi is None:
            continue
        res_name = mi.GetResidueName().strip()
        atom_name = mi.GetName().strip()

        # 跳过水
        if res_name in ('HOH', 'WAT', 'DOD'):
            continue

        if elem == 'H':
            # 检查是否属于 H-bond 供体原子
            for nb in atom.GetNeighbors():
                nb_mi = nb.GetMonomerInfo()
                if nb_mi is None:
                    continue
                nb_name = nb_mi.GetName().strip()
                if nb_name in HBOND_DONOR_ATOMS:
                    prot_donor_pairs.append((nb.GetIdx(), pos))
                    break
        else:
            # 重原子
            h_idx = len(prot_heavy_pos)  # 当前重原子在数组中的位置
            prot_heavy_pos.append(pos)
            prot_info.append({
                'residue': res_name,
                'atom_name': atom_name,
                'element': elem,
                'rdkit_idx': idx,
            })

            # 记录受体原子索引
            if atom_name in HBOND_ACCEPTOR_ATOMS:
                prot_acceptor_indices.append(h_idx)

            # 记录芳香残基的环原子
            if res_name in _AROM_RING_DEFS and elem != 'H':
                res_key = (mi.GetChainId(), mi.GetResidueNumber(), res_name)
                if res_key not in arom_res_atoms:
                    arom_res_atoms[res_key] = {}
                ring_defs = _AROM_RING_DEFS[res_name]
                for ring_label, ring_atom_names in ring_defs.items():
                    if atom_name in ring_atom_names:
                        if ring_label not in arom_res_atoms[res_key]:
                            arom_res_atoms[res_key][ring_label] = []
                        arom_res_atoms[res_key][ring_label].append(pos)

    if len(prot_heavy_pos) == 0:
        return None

    prot_heavy_pos = np.array(prot_heavy_pos)

    # 构建蛋白芳香环列表（质心+法向量）
    prot_rings = []
    for res_key, ring_dict in arom_res_atoms.items():
        for ring_label, positions in ring_dict.items():
            if len(positions) >= 4:  # 至少4个原子才能定义平面
                pts = np.array(positions)
                prot_rings.append({
                    'centroid': pts.mean(axis=0),
                    'normal': _compute_plane_normal(pts),
                    'positions': pts,
                })

    # ========== 3. 相互作用计数 ==========
    hydrophobic   = 0
    hbond_donor   = 0
    hbond_acceptor = 0
    pi_stacking   = 0
    salt_bridge   = 0
    metal_coord   = 0

    # --- 3a. 重原子对距离检查：疏水、盐桥、金属 ---
    for i, lig_pos in enumerate(lig_heavy_pos):
        lig_atom = lig_heavy_atoms[i]
        lig_elem = lig_atom.GetSymbol().upper()

        diffs = prot_heavy_pos - lig_pos
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))

        for j, dist in enumerate(dists):
            info = prot_info[j]
            res  = info['residue']
            elem_j = info['element']

            # 疏水接触
            if (dist <= CUTOFF_HYDROPHOBIC and
                res in HYDROPHOBIC_RES and
                lig_elem == 'C'):
                hydrophobic += 1

            # 盐桥
            if dist <= CUTOFF_SALT:
                if res in POSITIVE_RES and lig_elem in ('O', 'F', 'CL'):
                    salt_bridge += 1
                if res in NEGATIVE_RES and lig_elem == 'N':
                    salt_bridge += 1

            # 金属配位（增加：配体原子必须是 N,O,S）
            if (dist <= CUTOFF_METAL and
                elem_j in METAL_ELEMENTS and
                lig_elem in ('N', 'O', 'S')):
                metal_coord += 1

    # --- 3b. 氢键供体（蛋白供体 → 配体受体） ---
    for donor_idx, h_pos in prot_donor_pairs:
        donor_pos = np.array(conf_prot.GetAtomPosition(donor_idx))
        dh_vec = h_pos - donor_pos  # D→H 方向
        for i, lig_pos in enumerate(lig_heavy_pos):
            lig_atom = lig_heavy_atoms[i]
            if lig_atom.GetSymbol() not in ('O', 'N', 'F'):
                continue
            lig_pos = lig_heavy_pos[i]
            ha_dist = np.linalg.norm(h_pos - lig_pos)      # H···A 距离
            da_dist = np.linalg.norm(donor_pos - lig_pos)  # D···A 距离
            if da_dist <= CUTOFF_HBOND and ha_dist <= CUTOFF_HBOND_HA:
                ha_vec = lig_pos - h_pos  # H→A 方向
                dev = _angle_between(dh_vec, ha_vec)  # 偏离线性角度，0°=完美线性
                if dev <= CUTOFF_HBOND_DEV:
                    hbond_donor += 1

    # --- 3c. 氢键受体（配体供体 → 蛋白受体） ---
    for donor_idx, h_pos in lig_donor_pairs:
        donor_pos = np.array(conf_lig.GetAtomPosition(donor_idx))
        dh_vec = h_pos - donor_pos
        for acc_idx in prot_acceptor_indices:
            acc_pos = prot_heavy_pos[acc_idx]
            ha_dist = np.linalg.norm(h_pos - acc_pos)
            da_dist = np.linalg.norm(donor_pos - acc_pos)
            if da_dist <= CUTOFF_HBOND and ha_dist <= CUTOFF_HBOND_HA:
                ha_vec = acc_pos - h_pos
                dev = _angle_between(dh_vec, ha_vec)
                if dev <= CUTOFF_HBOND_DEV:
                    hbond_acceptor += 1

    # --- 3d. π-π 堆积（环质心 + 平面夹角） ---
    for pr in prot_rings:
        for lr in lig_rings:
            centroid_dist = np.linalg.norm(pr['centroid'] - lr['centroid'])
            if centroid_dist <= CUTOFF_PI_CENTROID:
                # 环平面夹角
                plane_angle = _angle_between(pr['normal'], lr['normal'])
                # 法向量夹角可能 > 90°，取锐角表示环平面夹角
                if plane_angle > 90:
                    plane_angle = 180 - plane_angle
                # 面-面堆积 (≤30°) 或 边-面堆积 (≥60°)
                if plane_angle <= 30.0 or plane_angle >= 60.0:
                    pi_stacking += 1

    # ========== 4. 归一化 ==========
    fp = np.array([hydrophobic, hbond_donor, hbond_acceptor,
                   pi_stacking, salt_bridge, metal_coord], dtype=float)
    fp = fp / max(n_lig_heavy, 1)

    return fp
