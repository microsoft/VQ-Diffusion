if [ -f cc_learnable.pth ]; then
    echo "cc_learnable.pth exists"
else
    echo "Downloading cc_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cc_learnable_ag
    
    cat cc_learnable_* > cc_learnable.pth
    rm cc_learnable_*
fi

if [ -f CC_pretrained.pth ]; then
    echo "CC_pretrained.pth exists"
else
    echo "Downloading cc_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/CC_pretrained_ag
    
    cat CC_pretrained_* > CC_pretrained.pth
    rm CC_pretrained_*
fi

if [ -f coco_learnable.pth ]; then
    echo "coco_learnable.pth exists"
else
    echo "Downloading coco_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_learnable_ag
    
    cat coco_learnable_* > coco_learnable.pth
    rm coco_learnable_*
fi

if [ -f coco_pretrained.pth ]; then
    echo "coco_pretrained.pth exists"
else
    echo "Downloading coco_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ag
    
    cat coco_pretrained_* > coco_pretrained.pth
    rm coco_pretrained_*
fi

if [ -f cub_pretrained.pth ]; then
    echo "cub_pretrained.pth exists"
else
    echo "Downloading cub_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/cub_pretrained_ag
    
    cat cub_pretrained_* > cub_pretrained.pth
    rm cub_pretrained_*
fi

if [ -f human_pretrained.pth ]; then
    echo "human_pretrained.pth exists"
else
    echo "Downloading human_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/human_pretrained_ag
    
    cat human_pretrained_* > human_pretrained.pth
    rm human_pretrained_*
fi

if [ -f imagenet_learnable.pth ]; then
    echo "imagenet_learnable.pth exists"
else
    echo "Downloading imagenet_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ag
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ah
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_learnable_ai
    
    cat imagenet_learnable_* > imagenet_learnable.pth
    rm imagenet_learnable_*
fi

if [ -f imagenet_pretrained.pth ]; then
    echo "imagenet_pretrained.pth exists"
else
    echo "Downloading imagenet_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ag
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ah
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/imagenet_pretrained_ai
    
    cat imagenet_pretrained_* > imagenet_pretrained.pth
    rm imagenet_pretrained_*
fi

if [ -f ithq_learnable.pth ]; then
    echo "ithq_learnable.pth exists"
else
    echo "Downloading ithq_learnable.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ag
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ah
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ai
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_aj
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_learnable_ak
    
    cat ithq_learnable_* > ithq_learnable.pth
    rm ithq_learnable_*
fi

if [ -f vqgan_ffhq_f16_1024.pth ]; then
    echo "vqgan_ffhq_f16_1024.pth exists"
else
    echo "Downloading vqgan_ffhq_f16_1024.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_aa
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ab
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ac
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ad
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ae
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_af
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ag
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ah
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_ai
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_ffhq_f16_1024_aj
    
    cat vqgan_ffhq_f16_1024_* > vqgan_ffhq_f16_1024.pth
    rm vqgan_ffhq_f16_1024_*
fi

if [ -f ithq_vqvae.pth ]; then
    echo "ithq_vqvae.pth exists"
else
    echo "Downloading ithq_vqvae.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_vqvae.pth
fi

if [ -f taming_f8_8192_openimages_last.pth ]; then
    echo "taming_f8_8192_openimages_last.pth exists"
else
    echo "Downloading taming_f8_8192_openimages_last.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/taming_f8_8192_openimages_last.pth
fi

if [ -f vqgan_imagenet_f16_16384.pth ]; then
    echo "vqgan_imagenet_f16_16384.pth exists"
else
    echo "Downloading vqgan_imagenet_f16_16384.pth"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/vqgan_imagenet_f16_16384.pth
fi

if [ -f ViT-B-32.pt ]; then
    echo "ViT-B-32.pt exists"
else
    echo "Downloading ViT-B-32.pt"
    wget https://github.com/tzco/storage/releases/download/vqdiffusion/ViT-B-32.pt
fi

