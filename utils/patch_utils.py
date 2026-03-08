import torch


def split_into_patches(img_tensor, patch_size=64):

    patches = []
    c, h, w = img_tensor.shape

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):

            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]

            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                padded = torch.zeros((c, patch_size, patch_size))
                padded[:, :patch.shape[1], :patch.shape[2]] = patch
                patch = padded

            patches.append(patch)

    return patches


def reconstruct_image(patches, original_shape, patch_size=64):

    c, h, w = original_shape
    restored = torch.zeros((c, h, w))

    idx = 0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):

            patch = patches[idx]

            restored[:, i:i+patch_size, j:j+patch_size] = patch[:, :min(patch_size, h-i), :min(patch_size, w-j)]

            idx += 1

    return restored


def restore_large_image(model, img_tensor, patch_size=64):

    patches = split_into_patches(img_tensor, patch_size)

    restored_patches = []

    for patch in patches:

        patch = patch.unsqueeze(0)

        with torch.no_grad():
            restored = model(patch)

        restored_patches.append(restored.squeeze(0))

    restored_img = reconstruct_image(restored_patches, img_tensor.shape, patch_size)

    return restored_img