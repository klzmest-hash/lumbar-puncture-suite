import os

import numpy as np
import torch.utils.data
import torch.nn.functional as F
import nibabel as nib
from scipy import ndimage


class Dataset3D(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, input_size=None, class_mapping=None):
        """
        3D Dataset for volume data (e.g., NIfTI files)
        
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            mask_ext (str): Mask file extension (e.g., '.nii', '.nii.gz').
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms. Defaults to None.
            input_size (int or tuple): Target size for resizing volumes. 
                                      If int, assumes cubic (D, H, W) = (size, size, size).
                                      If tuple, should be (D, H, W) or (H, W) (assumes D=H).
            class_mapping (dict, optional): Mapping from original class values to [0, num_classes-1].

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── case1.nii.gz
            │   ├── case2.nii.gz
            │   ├── ...
            |
            └── masks
                ├── case1.nii.gz
                ├── case2.nii.gz
                ├── ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        
        # Handle input_size
        if input_size is None:
            self.input_size = None
        elif isinstance(input_size, int):
            self.input_size = (input_size, input_size, input_size)
        elif isinstance(input_size, (list, tuple)):
            if len(input_size) == 2:
                self.input_size = (input_size[0], input_size[0], input_size[1])
            elif len(input_size) == 3:
                self.input_size = tuple(input_size)
            else:
                raise ValueError(f"input_size must be int, 2-tuple, or 3-tuple, got {input_size}")
        else:
            self.input_size = None
        
        # If provided external mapping table, use it; otherwise auto-create
        if class_mapping is not None:
            self.class_mapping = class_mapping
            print(f"✅ 使用外部提供的类别值映射表 (共{len(class_mapping)}个映射)")
        else:
            # Auto-create class mapping
            # Since all files have the same mapping, we can scan just one file
            self.class_mapping = self._create_class_mapping()

    def _create_class_mapping(self):
        """
        Scan mask files to auto-create class value mapping
        Maps actual mask values to continuous range [0, num_classes-1]
        
        Since all files typically have the same class values, we only scan the first file
        for faster initialization.
        """
        print(f"正在扫描mask文件以创建类别值映射...")
        all_unique_values = set()
        
        # Since all files have the same mapping, we only need to scan ONE file
        # This dramatically speeds up initialization
        SCAN_LIMIT = 1  # Only scan the first file
        
        files_to_scan = self.img_ids[:SCAN_LIMIT]
        
        print(f"   ⚡ 快速模式: 只扫描第一个文件来创建映射（假设所有文件使用相同的类别值）")
        
        # Scan mask files
        for idx, img_id in enumerate(files_to_scan):
            mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
            if os.path.exists(mask_path):
                try:
                    print(f"   正在扫描: {img_id}")
                    mask = nib.load(mask_path).get_fdata()
                    if mask is not None:
                        print(f"     文件大小: {mask.shape}, 数据类型: {mask.dtype}")
                        unique_vals = np.unique(mask)
                        all_unique_values.update(unique_vals.tolist())
                        print(f"     发现唯一值: {sorted(unique_vals.tolist())}")
                except Exception as e:
                    print(f"⚠️  读取mask文件失败: {mask_path}, 错误: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  mask文件不存在: {mask_path}")
        
        if len(all_unique_values) == 0:
            print("⚠️  警告: 未找到任何mask文件，将使用默认映射")
            return None
        
        # Sort all unique values
        sorted_values = sorted(all_unique_values)
        print(f"发现mask中的唯一值: {sorted_values}")
        print(f"值范围: [{min(sorted_values)}, {max(sorted_values)}]")
        
        # Check if mapping is needed
        max_val = max(sorted_values)
        min_val = min(sorted_values)
        
        # If values are already in valid range and continuous, no mapping needed
        if min_val >= 0 and max_val < self.num_classes and len(sorted_values) == max_val - min_val + 1:
            print(f"✅ mask值已在有效范围内 [0, {self.num_classes-1}]，无需映射")
            return None
        
        # Create mapping: map actual values to continuous range [0, len(unique_values)-1]
        mapping = {}
        num_actual_classes = min(len(sorted_values), self.num_classes)
        
        for i, orig_val in enumerate(sorted_values[:num_actual_classes]):
            mapping[int(orig_val)] = i
        
        print(f"📋 创建类别值映射表 (共{len(mapping)}个映射):")
        for orig_val, new_val in sorted(mapping.items()):
            print(f"   {orig_val} -> {new_val}")
        
        if len(sorted_values) > self.num_classes:
            print(f"⚠️  警告: mask中有{len(sorted_values)}个唯一值，但num_classes={self.num_classes}")
            print(f"   将只映射前{self.num_classes}个值，其余值将被忽略")
        
        return mapping
    
    def __len__(self):
        return len(self.img_ids)

    def _load_nifti(self, path):
        """Load NIfTI file and return numpy array"""
        try:
            nii = nib.load(path)
            data = nii.get_fdata()
            return data.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _resize_volume(self, volume, target_size, order=1):
        """
        Resize 3D volume to target size
        
        Args:
            volume: 3D numpy array (D, H, W)
            target_size: tuple (D, H, W)
            order: interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic)
        
        Returns:
            Resized volume
        """
        if volume.shape == target_size:
            return volume
        
        zoom_factors = (
            target_size[0] / volume.shape[0],
            target_size[1] / volume.shape[1],
            target_size[2] / volume.shape[2]
        )
        resized = ndimage.zoom(volume, zoom_factors, order=order, mode='nearest')
        return resized

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Load image and mask
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        
        image = self._load_nifti(img_path)
        mask = self._load_nifti(mask_path)
        
        if image is None or mask is None:
            raise ValueError(f"Failed to load image or mask for {img_id}")
        
        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Resize if needed
        if self.input_size is not None:
            # Resize can be slow for large 3D volumes, but necessary
            # For 512x512x183 -> 64x128x128, this will take some time
            image = self._resize_volume(image, self.input_size, order=1)  # Linear interpolation for image
            mask = self._resize_volume(mask, self.input_size, order=0)   # Nearest neighbor for mask
        
        # Apply transform if provided
        if self.transform is not None:
            # Transform should handle both image and mask
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Apply class mapping if provided
        if self.class_mapping is not None:
            mask_mapped = np.zeros_like(mask)
            for orig_val, new_val in self.class_mapping.items():
                mask_mapped[mask == orig_val] = new_val
            mask = mask_mapped
        else:
            # Validate mask value range
            mask_min = mask.min()
            mask_max = mask.max()
            if mask_max >= self.num_classes or mask_min < 0:
                mask = np.clip(mask, 0, self.num_classes - 1)
                unique_values = np.unique(mask)
                print(f"⚠️  警告: {img_id} 的mask值超出范围 [{mask_min}, {mask_max}]，已裁剪到 [0, {self.num_classes-1}]")
                print(f"    原始唯一值: {sorted(unique_values.tolist())}")
        
        # Convert to tensors
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()
        
        # One-hot encode mask: (D, H, W) -> (num_classes, D, H, W)
        mask = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask = torch.moveaxis(mask, -1, 0).to(torch.float)  # (D, H, W, C) -> (C, D, H, W)

        sample = {'image': image, 'mask': mask, 'id': img_id}
        return sample
