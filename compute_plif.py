"""
纯 RDKit + Biopython 计算简化 PLIF（6维相互作用指纹）
[疏水接触, H键供体, H键受体, π-π堆积, 盐桥, 金属配位]
每个维度的计数除以配体原子数进行归一化
"""

import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser


def compute_simple_plif(protein_pdb_path, ligand_path):
    """
    计算简化蛋白-配体相互作用指纹（6维向量）
    """
    # --- 加载配体 ---
    if ligand_path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(ligand_path, removeHs=False)
    else:
        mol = Chem.MolFromMolFile(ligand_path, removeHs=False)
    if mol is None:
        return None

    mol = Chem.RemoveHs(mol)  # 去掉氢，使用重原子计算
    conf = mol.GetConformer()
    lig_atoms = list(mol.GetAtoms())
    lig_positions = []
    for atom in lig_atoms:
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        lig_positions.append(np.array([pos.x, pos.y, pos.z]))

    if len(lig_positions) == 0:
        return None

    lig_positions = np.array(lig_positions)

    # --- 加载蛋白，收集重原子 ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb_path)

    protein_positions = []       # [x,y,z] for each heavy atom
    protein_atoms_info = []      # {residue, atom_name, element}

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.get_id()[0]
                # 跳过水和纯溶剂
                resname = residue.get_resname().strip()
                if resname in ('HOH', 'WAT', 'DOD'):
                    continue
                for atom in residue:
                    elem = atom.element.strip().upper()
                    if elem == 'H':
                        continue  # 跳过氢
                    protein_positions.append(atom.get_coord())
                    protein_atoms_info.append({
                        'residue': resname,
                        'hetflag': hetflag,
                        'atom_name': atom.get_name().strip(),
                        'element': elem,
                    })

    protein_positions = np.array(protein_positions)

    # --- 分类常量 ---
    hydrophobic_res = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO', 'TYR', 'CYS'}
    positive_res = {'LYS', 'ARG', 'HIS'}
    negative_res = {'ASP', 'GLU'}
    aromatic_res = {'PHE', 'TYR', 'TRP', 'HIS'}
    metal_elements = {'ZN', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU'}
    # 蛋白 H-bond 供体重原子（连接有极性H的N/O）
    hbond_donor_atoms = {'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ',
                         'OG', 'OG1', 'OH', 'SG'}
    # 蛋白 H-bond 受体重原子（有孤对电子的O/N）
    hbond_acceptor_atoms = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
                            'ND1', 'NE2', 'SD', 'SG'}

    # --- 截断半径 ---
    cutoff_hydrophobic = 4.5
    cutoff_hbond = 3.5
    cutoff_pi = 5.0
    cutoff_salt = 4.0
    cutoff_metal = 3.0

    # --- 计数 ---
    hydrophobic = 0
    hbond_donor = 0
    hbond_acceptor = 0
    pi_stacking = 0
    salt_bridge = 0
    metal_coord = 0

    for i, lig_pos in enumerate(lig_positions):
        lig_atom = lig_atoms[i]
        lig_elem = lig_atom.GetSymbol().upper()
        lig_aromatic = lig_atom.GetIsAromatic()

        # 计算与所有蛋白原子的距离
        diffs = protein_positions - lig_pos
        distances = np.sqrt(np.sum(diffs * diffs, axis=1))

        for j, dist in enumerate(distances):
            info = protein_atoms_info[j]
            res = info['residue']
            elem = info['element']
            atom_name = info['atom_name']
            is_het = info['hetflag'] != ' '

            # 1. 疏水接触：疏水残基的碳原子与配体碳原子
            if dist <= cutoff_hydrophobic and res in hydrophobic_res and lig_elem == 'C':
                hydrophobic += 1

            # 2. H键供体：蛋白供体 + 配体受体
            if dist <= cutoff_hbond and atom_name in hbond_donor_atoms and lig_elem in ('O', 'N', 'F'):
                hbond_donor += 1

            # 3. H键受体：蛋白受体 + 配体供体
            if dist <= cutoff_hbond and atom_name in hbond_acceptor_atoms and lig_elem in ('O', 'N'):
                hbond_acceptor += 1

            # 4. π-π堆积：芳香残基 + 芳香配体原子
            if dist <= cutoff_pi and res in aromatic_res and lig_aromatic:
                pi_stacking += 1

            # 5. 盐桥：带正电残基 + 配体负电原子 / 带负电残基 + 配体正电原子
            if dist <= cutoff_salt:
                if res in positive_res and lig_elem in ('O', 'F', 'CL'):
                    salt_bridge += 1
                if res in negative_res and lig_elem in ('N'):
                    salt_bridge += 1

            # 6. 金属配位
            if dist <= cutoff_metal and elem in metal_elements:
                metal_coord += 1

    # --- 归一化 ---
    n_lig = len(lig_positions)
    fp = np.array([hydrophobic, hbond_donor, hbond_acceptor,
                   pi_stacking, salt_bridge, metal_coord], dtype=float)
    fp = fp / max(n_lig, 1)

    return fp
