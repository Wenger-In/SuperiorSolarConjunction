import matplotlib
import shutil
import os

# 获取matplotlib字体目录
matplotlib_font_dir = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
print(f"Matplotlib字体目录: {matplotlib_font_dir}")

# 源字体文件路径
source_font = r'C:\Fonts_add\Helvetica.ttf'  # 确保路径正确

# 目标字体文件路径
target_font = os.path.join(matplotlib_font_dir, 'Helvetica.ttf')

# 复制字体文件
if os.path.exists(source_font):
    shutil.copy2(source_font, target_font)
    print(f"已将Helvetica字体复制到: {target_font}")
    
    # 删除字体缓存
    cache_dir = matplotlib.get_cachedir()
    cache_file = os.path.join(cache_dir, 'fontlist-v330.json')  # 文件名可能因版本而异
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"已删除字体缓存文件: {cache_file}")
        print("请重启Python内核以使更改生效。")
else:
    print(f"源字体文件不存在: {source_font}")