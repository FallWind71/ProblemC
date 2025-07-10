import matplotlib.font_manager as fm

# 获取所有可用字体路径
font_paths = fm.findSystemFonts()

font_names = []
for path in font_paths:
    try:
        font_prop = fm.FontProperties(fname=path)
        name = font_prop.get_name()
        font_names.append(name)
    except Exception:
        # 遇到错误字体就跳过
        continue

# 去重并排序
font_names = sorted(set(font_names))

print("所有可用字体:")
for f in font_names:
    print(f)

# 过滤可能的中文字体
chinese_fonts = [f for f in font_names if any(kw in f for kw in ['Sim', 'Kai', 'Hei', 'Fang', 'Song', 'YaHei', 'Noto', '思源', '文泉', '黑体', '楷体', '仿宋'])]
print("\n可能的中文字体:")
for f in chinese_fonts:
    print(f)
