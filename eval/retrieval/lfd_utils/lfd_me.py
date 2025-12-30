# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
'''
Function is modified based on https://github.com/kacperkan/light-field-distance
'''
import argparse
import sys
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

SIMILARITY_TAG = b"SIMILARITY:"
import lfd
CURRENT_DIR = Path(lfd.__file__).parent / "Executable"
if not CURRENT_DIR.exists():
    raise FileNotFoundError(f"[ERROR] LFD Executable folder not found: {CURRENT_DIR}")

GENERATED_FILES_NAMES = [
    "all_q4_v1.8.art",
    "all_q8_v1.8.art",
    "all_q8_v1.8.cir",
    "all_q8_v1.8.ecc",
    "all_q8_v1.8.fd",
]

OUTPUT_NAME_TEMPLATES = [
    "{}_q4_v1.8.art",
    "{}_q8_v1.8.art",
    "{}_q8_v1.8.cir",
    "{}_q8_v1.8.ecc",
    "{}_q8_v1.8.fd",
]


class MeshEncoder:
    """Class holding an object and preprocessing it using an external cmd."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, folder=None, file_name=None):
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        if folder is None:
            folder = tempfile.mkdtemp()
        if file_name is None:
            file_name = uuid.uuid4()
        self.temp_dir_path = Path(folder)
        self.file_name = file_name
        self.temp_path = self.temp_dir_path / "{}.obj".format(self.file_name)
        self.mesh.export(self.temp_path.as_posix())

    def get_path(self) -> str:
        return self.temp_path.with_suffix("").as_posix()

    def align_mesh(self):
        """Create data of a 3D mesh to calculate Light Field Distance."""
        run_dir_path = self.temp_dir_path # self.temp_dir_path is already a Path object from __init__
        
        # Ensure these are defined in lfd_me.py or passed appropriately
        # Example:
        # copy_file_names = ['3DAlignment', 'align10.txt', 'q8_table', '12_0.obj', ..., '12_9.obj']
        copy_file_names = ['3DAlignment', 'align10.txt', 'q8_table'] + [f'12_{i}.obj' for i in range(10)]


        copied_file_paths = []

        try:
            for f_name in copy_file_names:
                src_path = CURRENT_DIR / f_name
                dst_path = run_dir_path / f_name
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_file_paths.append(dst_path)
                except Exception as e:
                    print(f"[ERROR] Failed to copy {src_path} to {dst_path}: {e}")
                    raise

            # Run the 3DAlignment executable
            alignment_executable = run_dir_path / '3DAlignment'
            # Ensure it's executable (shutil.copy2 should preserve permissions, but good to be sure on some systems)
            # os.chmod(alignment_executable, 0o755) # Might be needed if permissions are an issue

            cmd = [str(alignment_executable), self.temp_path.with_suffix("").as_posix()]
            
            process = subprocess.Popen(
                cmd,
                cwd=str(run_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_message = (
                    f"3DAlignment failed for {self.temp_path.as_posix()} with return code {process.returncode}.\n"
                    f"STDERR: {stderr}\n"
                    f"STDOUT: {stdout}"
                )
                print(f"[ERROR] {error_message}")
                raise RuntimeError(error_message)

            # Move/Rename generated files
            for gen_file_name_str, out_file_template_str in zip(GENERATED_FILES_NAMES, OUTPUT_NAME_TEMPLATES):
                # GENERATED_FILES_NAMES are like "all_q4_v1.8.art"
                # OUTPUT_NAME_TEMPLATES are like "{}_q4_v1.8.art"
                src_generated_file = run_dir_path / gen_file_name_str
                
                # self.file_name is passed during MeshEncoder instantiation (e.g., "mesh_q4_v1.8" or "some_model_name")
                final_name_for_this_file = out_file_template_str.format(self.file_name)
                dst_final_path = run_dir_path / final_name_for_this_file

                if src_generated_file.exists():
                    shutil.move(str(src_generated_file), str(dst_final_path))
                else:
                    # This is a critical error if an expected file by 3DAlignment is not produced
                    missing_file_error = f"Expected output file {src_generated_file} not found after 3DAlignment execution."
                    print(f"[ERROR] {missing_file_error}")
                    # If any of the primary outputs (like *_q4_v1.8.art) is missing, it's a failure.
                    if "_q4_v1.8.art" in gen_file_name_str: # Check if it's one of the main .art files
                         raise FileNotFoundError(missing_file_error)


        finally:
            # Guaranteed cleanup of copied files
            for f_path in copied_file_paths:
                if f_path.exists():
                    try:
                        os.remove(f_path)
                    except OSError as e_rm:
                        print(f"[WARN] Failed to remove copied temporary file {f_path}: {e_rm}")
            
            # This file is created by MeshEncoder's __init__
            if self.temp_path.exists():
                try:
                    os.remove(self.temp_path)
                except OSError as e_rm:
                    print(f"[WARN] Failed to remove temporary mesh file {self.temp_path}: {e_rm}")
