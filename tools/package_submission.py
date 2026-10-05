"""Create three independent, compact submission ZIPs without expanded copies."""
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ('TT1_BINARY_SEARCH_GIAI_THICH.md', 'TT2_HISTOGRAM_GIAI_THICH.md',
          'TT3_GAUSSIAN_GIAI_THICH.md')

def build_package(number):
    key = f'tt{number}'
    target = ROOT / 'submission' / f'THUAT_TOAN_{number}.zip'
    prefix = f'THUAT_TOAN_{number}/'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        package_paths = {f'main_{key}.py': 'main.py',
                         f'reports/{GUIDES[number - 1]}': 'GIAI_THICH_THUAT_TOAN.md',
                         'reports/KET_QUA_HIEN_TAI.md': 'KET_QUA_HIEN_TAI.md'}
        def add(path, member):
            archive.write(path, prefix + member)
            package_paths[path.relative_to(ROOT).as_posix()] = member
        for folder in ('app', 'core', 'algorithms'):
            for path in sorted((ROOT / folder).rglob('*.py')):
                add(path, path.relative_to(ROOT).as_posix())
        main = (ROOT / f'main_{key}.py').read_text(encoding='utf-8').replace(f'main_{key}.py', 'main.py')
        archive.writestr(prefix + 'main.py', main)
        add(ROOT / 'requirements.txt', 'requirements.txt')
        for extension in ('pptx', 'pdf'):
            filename = f'THUAT_TOAN_{number}.{extension}'
            add(ROOT / 'slides' / f'THUAT_TOAN_{number}' / filename, filename)
        for filename in ('test_metrics.csv', 'summary.csv', 'test_thresholds.csv', 'run_config.json'):
            add(ROOT / 'outputs/tables' / key / filename, 'evidence/' + filename)
        add(ROOT / 'outputs/models' / f'{key}.json', 'evidence/model.json')
        for stem in ('phone_F2', 'phone_M2', 'studio_F2', 'studio_M2'):
            add(ROOT / 'outputs/figures' / key / f'{stem}.png', 'evidence/' + stem + '.png')
            add(ROOT / 'outputs/diagnostics' / key / f'{stem}.json',
                'evidence/diagnostics/' + stem + '.json')
        for folder, filename, member in (
            ('all', 'test_metrics.csv', 'four_test_metrics.csv'),
            ('all_all_dataset', 'test_metrics.csv', 'all_dataset_metrics.csv'),
            ('all_all_dataset', 'summary.csv', 'all_dataset_summary.csv')):
            add(ROOT / 'outputs/tables' / folder / filename, 'evidence/comparison/' + member)
        if number == 3:
            add(ROOT / 'outputs/figures/training/gaussian_distributions.png',
                'evidence/gaussian_distributions.png')
        if number == 2:
            for filename in ('sweep.csv', 'summary.csv', 'best_by_file.csv', 'selection.json'):
                add(ROOT / 'outputs/tables/tt2_w_selection' / filename, 'evidence/w_selection/' + filename)
        for split in ('train', 'test'):
            archive.writestr(prefix + f'data/{split}/README.md',
                f'Đặt 4 WAV và 4 LAB cùng tên của tập {split} tại đây. ZIP không chứa WAV theo quy định nộp bài.\n')
        def portable_document(filename):
            document = ROOT / 'reports' / filename
            content = document.read_text(encoding='utf-8')
            def replace_link(match):
                label, source = match.groups()
                source = source.strip('<>')
                if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', source) and not source.startswith('H:/GIUAKI_XLTHS/'):
                    return match.group(0)
                if source.startswith('#'):
                    return match.group(0)
                code_path, separator, anchor = source.partition('#')
                legacy_path, legacy_separator, line = code_path.rpartition(':')
                if legacy_separator and line.isdigit():
                    code_path, anchor = legacy_path, f'L{line}'
                if code_path.startswith('H:/GIUAKI_XLTHS/'):
                    code_path = code_path[len('H:/GIUAKI_XLTHS/'):]
                else:
                    try:
                        code_path = (document.parent / code_path).resolve().relative_to(ROOT).as_posix()
                    except ValueError:
                        return label
                member = package_paths.get(code_path)
                if member is None:
                    return label  # References to other algorithms remain plain text.
                fragment = f'#{anchor}' if anchor else ''
                return f'[{label}]({member}{fragment})'
            return re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', replace_link, content)
        archive.writestr(prefix + 'GIAI_THICH_THUAT_TOAN.md', portable_document(GUIDES[number - 1]))
        archive.writestr(prefix + 'KET_QUA_HIEN_TAI.md', portable_document('KET_QUA_HIEN_TAI.md'))
        special = ('TT2 chỉ Histogram Energy; không dùng Spectral Centroid. W = 20 chung; W nguyên 1–50 '
                   'đồng tối ưu trên 4 test. LAB test tham gia chọn W, nên kết quả là test-tuned.\n'
                   if number == 2 else 'Tham số thuật toán và noise statistics học từ TRAIN.\n')
        archive.writestr(prefix + 'README.md', f'''# Thuật toán {number} — gói nộp

`main.py` cố định TT{number}. Giải nén ZIP, bổ sung dữ liệu gốc vào `data/train/` và `data/test/`, rồi chạy:

```powershell
python -m pip install -r requirements.txt
python main.py
python main.py --file phone_F2.wav
python main.py --no-show
python main.py --evaluate-all
```

Cần Python ≥ 3.10, có Tkinter để mở cửa sổ. Mặc định chạy 4 test, frame 25 ms/hop 10 ms; HIGH xác nhận, LOW duy trì; nối silence < 200 ms, bỏ speech < 100 ms. MAE chỉ dùng START/END của FINAL regions.

{special}
`GIAI_THICH_THUAT_TOAN.md` giải thích riêng thuật toán này bằng tiếng Việt. PPTX/PDF gồm 7 slide tiếng Anh, biểu đồ/bảng chỉnh sửa được trong PPTX. `evidence/` chứa kết quả 4 test. ZIP không có WAV.

Tên/MSSV còn chờ bổ sung. Trước nộp đổi tên thư mục thành `MaTheSV-HoTen` và thay thông tin pending trên slide đầu. Thuyết trình khoảng 3 phút, demo 1 phút.

Module chung có cả ba thuật toán do pipeline cần import; sinh viên chạy phần mình bằng main cố định. Link tài liệu trong ZIP dùng đường dẫn tương đối để đọc trên máy khác.
''')
        members = archive.namelist()
        assert not any(Path(name).suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg') for name in members)
        assert not any('__pycache__' in name for name in members)
    print(f'TT{number}: {len(members)} files, no WAV; {target.name}')

def main():
    (ROOT / 'submission').mkdir(exist_ok=True)
    for number in range(1, 4):
        build_package(number)
    (ROOT / 'submission/README.md').write_text('''# Ba gói nộp độc lập

Chỉ giữ ba ZIP hiện hành: `THUAT_TOAN_1.zip`, `THUAT_TOAN_2.zip`, `THUAT_TOAN_3.zip`.

Mỗi ZIP có main cố định, module chung, hướng dẫn riêng thuật toán, kết quả và slide tiếng Anh. Không có WAV. Giải nén ZIP rồi thêm dữ liệu gốc để chạy trên máy cá nhân. Họ tên/MSSV bổ sung sau; đổi thư mục thành `MaTheSV-HoTen` trước nộp.
''', encoding='utf-8')

if __name__ == '__main__':
    main()
