import smplx

model = smplx.create(
    model_path="models",
    model_type="smplx",
    gender="neutral"
)
print("✅ SMPL-X模型加载成功！")