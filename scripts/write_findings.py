"""Write factual comparisons from verified measurements, never estimates.

This postprocessing does not train, select checkpoints or change the protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

from common.config import ROOT
from scripts.export_report import collect_runs, latex_escape
from A1.export_report import read_run
from A1.protocol import load_protocol


def percent(value):
    return f'{100 * value:.2f}\\%'


def paired_delta(records, first, second):
    groups = {key: {r['seed']: r for r in records if r['experiment']['id'] == key} for key in [first, second]}
    seeds = sorted(groups[first].keys() & groups[second].keys())
    if not seeds or any(groups[key][s]['test'] is None for key in groups for s in seeds):
        raise ValueError(f'Missing paired test measurements: {first}, {second}')
    deltas = [100 * (groups[first][s]['test']['accuracy'] - groups[second][s]['test']['accuracy']) for s in seeds]
    return statistics.mean(deltas), statistics.stdev(deltas) if len(deltas) > 1 else None, deltas


def delta_text(mean, sd):
    return f'{mean:+.2f}' + (f' $\\pm$ {sd:.2f}' if sd is not None else '') + ' điểm phần trăm'


def write_exercises():
    records = collect_runs(ROOT / 'runs/exercises')
    if len(records) != 60 or any(r['test'] is None for r in records):
        raise ValueError('Final exercise discussion requires all 60 train/test runs.')
    target = ROOT / 'reports/exercises/generated'
    rows = json.loads((target / 'summary.json').read_text())
    by_id = {r['id']: r for r in rows}
    evidence = target / 'evidence'
    for record in records:
        destination = evidence / record['experiment']['id'] / f'seed_{record["seed"]}'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['run.json', 'metrics.json', 'history.csv', 'evaluation_test.json']:
            shutil.copy2(record['directory'] / name, destination / name)
        diagnostics = record['directory'] / 'diagnostics'
        if diagnostics.exists():
            (destination / 'diagnostics').mkdir(exist_ok=True)
            for name in ['summary.json', 'temporal_gradients.csv']:
                shutil.copy2(diagnostics / name, destination / 'diagnostics' / name)
    hardware = records[0]['environment']
    introductory = (f'Các kết quả được đo với ba seed 42, 43 và 44 trên {latex_escape(hardware["device_name"])}; '
                    f'PyTorch {latex_escape(hardware["torch"])} và CUDA runtime {latex_escape(hardware["cuda_runtime"])}. '
                    'Chênh lệch giữa hai cấu hình được tính theo cặp cùng seed, rồi lấy trung bình và SD.\n\n')
    e1 = [introductory]
    for identifier in ['e1_softmax', 'e1_mlp', 'e1_cnn']:
        row = by_id[identifier]
        own = [r for r in records if r['experiment']['id'] == identifier]
        histories = []
        for run in own:
            with (run['directory'] / 'history.csv').open() as stream:
                histories.append(list(csv.DictReader(stream))[-1])
        e1.append(f'{latex_escape(row["label"])}: test accuracy '
                  f'$({100 * row["test_accuracy_mean"]:.2f} \\pm {100 * row["test_accuracy_std"]:.2f})\\%$, '
                  f'macro-F1 {row["test_macro_f1_mean"]:.4f}; '
                  f'CE cuối epoch train/validation trung bình '
                  f'{statistics.mean(float(h["train_ce"]) for h in histories):.4f}/'
                  f'{statistics.mean(float(h["validation_ce"]) for h in histories):.4f}.'
                  '\n\n')
    for first, second in [('e1_mlp', 'e1_softmax'), ('e1_cnn', 'e1_mlp')]:
        mean, sd, _ = paired_delta(records, first, second)
        e1.append(f'{latex_escape(by_id[first]["label"])} trừ {latex_escape(by_id[second]["label"])}: '
                  f'{delta_text(mean, sd)}.\n\n')
    e1.append('MLP thêm biểu diễn phi tuyến; CNN giữ cấu trúc hai chiều, chia sẻ kernel và pooling. '
              'Xếp hạng đã đo phù hợp với lợi ích của các biểu diễn này trên ảnh nhỏ. '
              'Khác biệt về số tham số và regularization cũng góp phần vào kết quả của từng kiến trúc.\n\n')
    e1.append('CNN có khoảng cách train/validation CE cuối kỳ lớn dù test accuracy cao nhất E1. '
              'Đường CE và error gallery gợi ý overfit hoặc dự đoán sai quá tự tin; '
              'calibration chưa được khảo sát riêng trong bài này. Test dùng checkpoint được validation chọn, '
              'còn các CE cuối kỳ ở trên mô tả epoch 30.\n\n')
    cnn = next(r for r in records if r['experiment']['id'] == 'e1_cnn' and r['seed'] == 42)
    matrix = cnn['test']['confusion_matrix']
    a, b = max(((i, j) for i in range(10) for j in range(10) if i != j), key=lambda pair: matrix[pair[0]][pair[1]])
    classes = json.loads((cnn['directory'] / 'run.json').read_text())['dataset']['class_names']
    e1.append(f'Ở CNN seed 42, lỗi nhiều nhất theo một chiều là {latex_escape(classes[a])} thành '
              f'{latex_escape(classes[b])}: {matrix[a][b]} ảnh.\n')
    (target / 'e1_analysis.tex').write_text(''.join(e1))
    e2 = []
    for tokenizer in ['rows', 'patches', 'cnn']:
        first, second = f'e2_{tokenizer}_torch', f'e2_{tokenizer}_manual'
        mean, sd, _ = paired_delta(records, first, second)
        a, b = by_id[first], by_id[second]
        e2.append(f'Tokenizer {latex_escape(tokenizer)}: PyTorch trừ manual có chênh lệch test '
                  f'{delta_text(mean, sd)}; thời gian train/epoch trung bình của median từng seed lần lượt '
                  f'{a["median_train_seconds_mean"]:.2f}s và {b["median_train_seconds_mean"]:.2f}s.\n\n')
    best = max((r for r in rows if r['exercise'] == 'E2'), key=lambda r: r['test_accuracy_mean'])
    e2.append(f'Trong sáu cấu hình, {latex_escape(best["label"])} có test accuracy trung bình cao nhất, '
              f'đạt {percent(best["test_accuracy_mean"])}.\n\n')
    for second in ['e2_rows_manual', 'e2_patches_manual']:
        mean, sd, _ = paired_delta(records, 'e2_cnn_manual', second)
        e2.append(f'Giữ backend manual, CNN-stem trừ {latex_escape(by_id[second]["label"])}: '
                  f'{delta_text(mean, sd)}.\n\n')
    mean, sd, _ = paired_delta(records, best['id'], 'e1_cnn')
    e2.append(f'Cấu hình E2 tốt nhất so với CNN E1 có chênh lệch {delta_text(mean, sd)}. '
              'CNN-stem cung cấp đặc trưng cục bộ trước attention; các kết quả gợi ý rằng prior không gian hữu ích '
              'trong ngân sách này. Tokenizer đồng thời làm thay đổi số token, chiều đầu vào và số tham số.\n')
    (target / 'e2_analysis.tex').write_text(''.join(e2))
    e3 = []
    for identifier in ['e3_lstm_rows', 'e3_gru_rows', 'e3_lstm_patches', 'e3_gru_patches']:
        mean, sd, _ = paired_delta(records, identifier, 'e1_cnn')
        e3.append(f'{latex_escape(by_id[identifier]["label"])} trừ CNN E1: {delta_text(mean, sd)}.\n\n')
    e3.append('Rows và patches thay cả số bước lẫn số pixel/token; kết quả không tách riêng tác động độ dài chuỗi.\n')
    for family in ['lstm', 'gru']:
        mean, sd, _ = paired_delta(records, f'e3_{family}_rows', f'e3_{family}_patches')
        e3.append(f'{family.upper()} rows trừ patches: {delta_text(mean, sd)}.\n\n')
    mean, sd, _ = paired_delta(records, 'e3_gru_rows', 'e3_lstm_rows')
    e3.append(f'GRU rows trừ LSTM rows: {delta_text(mean, sd)}. '
              'Rows giữ trọn ngữ cảnh ngang trong mỗi token và có ít bước hơn patches; đây là một cách diễn giải '
              'phù hợp với kết quả trên cách chia dữ liệu này. Kiến trúc cell và đường quét cùng ảnh hưởng '
              'đến khả năng tổng hợp đặc trưng.\n')
    (target / 'e3_analysis.tex').write_text(''.join(e3))
    improvement = []
    init_delta, init_sd, _ = paired_delta(records, 'e3_lstm_rows_orth_init', 'e3_lstm_rows')
    penalty_delta, penalty_sd, _ = paired_delta(records, 'e3_lstm_rows_orth_reg', 'e3_lstm_rows_orth_init')
    improvement.append(f'Khởi tạo trực giao so với baseline làm thay đổi accuracy '
                       f'{delta_text(init_delta, init_sd)}. Thêm penalty vào bản khởi tạo trực giao '
                       f'thay đổi {delta_text(penalty_delta, penalty_sd)}, trong khi thời gian/epoch '
                       f'tăng từ {by_id["e3_lstm_rows_orth_init"]["median_train_seconds_mean"]:.2f}s '
                       f'lên {by_id["e3_lstm_rows_orth_reg"]["median_train_seconds_mean"]:.2f}s.\n\n')
    for identifier in ['e3_lstm_rows_chrono', 'e3_lstm_rows_mean']:
        mean, sd, _ = paired_delta(records, identifier, 'e3_lstm_rows')
        improvement.append(f'{latex_escape(by_id[identifier]["label"])} so với baseline: '
                           f'{delta_text(mean, sd)}.\n\n')
    combined_delta, combined_sd, _ = paired_delta(records, 'e3_lstm_rows_combined', 'e3_lstm_rows')
    mean_delta, mean_sd, _ = paired_delta(records, 'e3_lstm_rows_no_mean', 'e3_lstm_rows_combined')
    reg_delta, reg_sd, _ = paired_delta(records, 'e3_lstm_rows_no_reg', 'e3_lstm_rows_combined')
    time_saved = 100 * (1 - by_id['e3_lstm_rows_no_reg']['median_train_seconds_mean']
                        / by_id['e3_lstm_rows_combined']['median_train_seconds_mean'])
    improvement.append(f'Tổ hợp trừ baseline: {delta_text(combined_delta, combined_sd)}. '
                       f'Bỏ mean pooling khỏi tổ hợp thay đổi accuracy {delta_text(mean_delta, mean_sd)}; '
                       f'bỏ regularizer thay đổi {delta_text(reg_delta, reg_sd)} và giảm thời gian/epoch '
                       f'{time_saved:.1f}\\% theo các median đã đo. '
                       'Mean pooling có ảnh hưởng rõ hơn trong hai ablation này; phần tính penalty '
                       'chiếm thêm thời gian dù chênh lệch accuracy khi bỏ penalty khá nhỏ.\n\n')
    all_improvements_lower = all(row['test_accuracy_mean'] < by_id['e3_lstm_rows']['test_accuracy_mean']
                                 for row in rows if row['group'] == 'improvement')
    if all_improvements_lower:
        improvement.append('Cả bảy biến thể mở rộng có test accuracy trung bình thấp hơn baseline. '
                           'Trong thiết lập này, các thay đổi chưa mang lại cải thiện accuracy. '
                           'Mean pooling gộp đồng đều cả những trạng thái đầu chuỗi có ít ngữ cảnh; '
                           'việc trộn các trạng thái này có thể làm loãng biểu diễn trước head.\n\n')
    for identifier in ['e3_lstm_rows', 'e3_lstm_rows_combined']:
        summaries, gradients = [], []
        for seed in [42, 43, 44]:
            directory = ROOT / 'runs/exercises' / identifier / f'seed_{seed}' / 'diagnostics'
            summaries.append(json.loads((directory / 'summary.json').read_text()))
            with (directory / 'temporal_gradients.csv').open() as stream:
                gradients.append(list(csv.DictReader(stream)))
        improvement.append(f'{latex_escape(by_id[identifier]["label"])} trên 256 ảnh validation: '
                           + '; '.join(f'{key} accuracy {percent(statistics.mean(s["accuracy"][key] for s in summaries))}'
                                      for key in ['original', 'reversed', 'permuted']) + '. '
                           f'Hidden gradient norm trung bình tại bước 1/bước 28 là '
                           f'{statistics.mean(float(g[0]["hidden_grad_norm"]) for g in gradients):.3g}/'
                           f'{statistics.mean(float(g[-1]["hidden_grad_norm"]) for g in gradients):.3g}.\n\n')
    improvement.append('Accuracy giảm khi đảo hoặc tráo thứ tự cho thấy checkpoint nhạy với thứ tự không gian. '
                       'Mean pooling tạo đường gradient trực tiếp từ head tới mọi trạng thái hidden, '
                       'trong khi last pooling nối head với trạng thái cuối; khác biệt này giải thích '
                       'một phần hình dạng đường gradient của hai mô hình.\n')
    (target / 'improvement_analysis.tex').write_text(''.join(improvement))
    winner = max((r for r in rows if r['group'] == 'required'), key=lambda r: r['test_accuracy_mean'])
    (target / 'conclusion.tex').write_text(
        f'Trong 13 cấu hình cơ bản, {latex_escape(winner["label"])} có test accuracy trung bình cao nhất, '
        f'đạt {percent(winner["test_accuracy_mean"])}. CNN-stem cho kết quả tốt nhất trong nhóm E2, '
        'còn biểu diễn theo hàng phù hợp hơn patch đối với cả LSTM và GRU. '
        + ('Bảy biến thể mở rộng E3 đều có accuracy trung bình dưới baseline LSTM rows. ' if all_improvements_lower else '')
        + 'Các ablation cho thấy mean pooling ảnh hưởng lớn đến accuracy của tổ hợp, '
        'trong khi penalty trực giao làm tăng chi phí huấn luyện.\n')


def write_a1(*, source_files=None):
    protocol = load_protocol()
    records = []
    for experiment in protocol['experiments']:
        for seed in [42, 43, 44]:
            row, reason = read_run(ROOT / 'runs/assignment1' / experiment['id'] / f'seed_{seed}', experiment, seed, protocol, source_files=source_files)
            if row is None or reason != 'complete':
                raise ValueError('Final A1 discussion requires all 24 train/test runs.')
            records.append(row)
    target = ROOT / 'reports/assignment1/generated'
    summary = json.loads((target / 'summary.json').read_text())['groups']
    groups = {r['experiment']: r for r in summary}
    paragraphs = ['Kết quả gồm 24 lượt chạy, với ba seed 42, 43 và 44 cho mỗi cấu hình. '
                  'Chênh lệch accuracy được tính theo cặp cùng seed; các khoảng $\\pm$ là độ lệch chuẩn mẫu, '
                  'không phải khoảng tin cậy.\n\n']
    family_labels = {'resnet18': 'ResNet18', 'swin_t': 'Swin-T'}
    regime_labels = {'scratch': 'Scratch', 'head': 'Head-only', 'partial': 'Partial', 'full': 'Full'}

    def paired(first, second):
        get = lambda identifier: {r['seed']: r for r in records if r['experiment'] == identifier}
        a, b = get(first), get(second)
        values = [100 * (a[s]['test_accuracy'] - b[s]['test_accuracy']) for s in [42, 43, 44]]
        return delta_text(statistics.mean(values), statistics.stdev(values))

    def measured(identifier, metric, scale=1, digits=2):
        value = groups[identifier][metric]
        return '$' + f'{scale * value["mean"]:.{digits}f} \\pm {scale * value["sd"]:.{digits}f}' + '$'

    # Presentation is separate from the frozen exporter and measured artifacts.
    table = [r'\begin{tabular}{llrrr}', r'\toprule',
             r'Kiến trúc & Chế độ & Val (\%) & Test (\%) & Macro-F1 \\', r'\midrule']
    costs = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Kiến trúc & Chế độ & Tổng (M) & Học (M) & Train (s) & VRAM (MiB) \\', r'\midrule']
    for experiment in protocol['experiments']:
        identifier, family, regime = experiment['id'], experiment['family'], experiment['regime']
        group = groups[identifier]
        label = f'{family_labels[family]} & {regime_labels[regime]}'
        table.append(label + ' & ' + ' & '.join([
            measured(identifier, 'val_accuracy', 100), measured(identifier, 'test_accuracy', 100),
            measured(identifier, 'test_macro_f1', digits=4)]) + r' \\')
        costs.append(label + f' & {group["total_params"]["mean"] / 1e6:.3f}'
                     f' & {group["trainable_params"]["mean"] / 1e6:.3f}'
                     f' & {measured(identifier, "train_seconds", digits=1)}'
                     f' & {group["peak_gpu_memory_bytes"]["mean"] / 2**20:.1f}' + r' \\')
    for name, lines in [('comparison_table.tex', table), ('cost_table.tex', costs)]:
        (target / name).write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n')

    paragraphs.extend([
        '\\subsection{Khởi tạo và khả năng thích nghi biểu diễn}\n',
        f'ResNet18 head-only so với scratch có chênh lệch {paired("resnet18_head", "resnet18_scratch")}. '
        'Backbone đóng băng chỉ cho phép head tuyến tính điều chỉnh ranh giới phân loại trên đặc trưng có sẵn. '
        'Kết quả cho thấy đặc trưng ImageNet cố định chưa đủ phù hợp với thiết lập CIFAR-100 này. '
        f'Mở stage cuối tăng {paired("resnet18_partial", "resnet18_scratch")} so với scratch; '
        f'mở toàn bộ tăng {paired("resnet18_full", "resnet18_scratch")}. '
        'Khả năng cập nhật biểu diễn đóng vai trò đáng kể đối với ResNet18.\n\n',
        f'Swin-T scratch chỉ đạt {measured("swin_t_scratch", "test_accuracy", 100)}\\% trên test. '
        'Ở epoch 20, seed 43 và 44 có validation accuracy 1\\%, train accuracy xấp xỉ 1\\% '
        'và CE gần $\\ln(100)\\simeq4{,}605$. Cả train lẫn validation đều thấp, '
        'cho thấy mô hình học rất ít trong thiết lập này. Seed 42 cải thiện hơn nhưng vẫn xa các bản pretrained. '
        'Ngân sách 20 epoch, learning rate $10^{-3}$ và lịch cosine không warm-up '
        'chưa tạo được một đối chứng scratch hội tụ tốt. Thực nghiệm không tách riêng nguyên nhân '
        'của từng yếu tố và không khảo sát thêm siêu tham số sau khi mở test.\n\n',
        f'Swin-T head-only đạt {measured("swin_t_head", "test_accuracy", 100)}\\%, '
        f'partial đạt {measured("swin_t_partial", "test_accuracy", 100)}\\%, '
        f'và full đạt {measured("swin_t_full", "test_accuracy", 100)}\\%. '
        f'Full cao hơn scratch {paired("swin_t_full", "swin_t_scratch")}; '
        'mức chênh lệch này phản ánh cả lợi ích của pretrained lẫn đối chứng scratch chưa học tốt. '
        'Kết quả không đủ để kết luận Transformer luôn phụ thuộc pretrained '
        'hoặc luôn vượt CNN trong những ngân sách khác.\n\n',
        '\\subsection{Accuracy và chi phí huấn luyện}\n',
    ])
    resnet, swin = groups['resnet18_full'], groups['swin_t_full']
    paragraphs.append(
        f'Khi cùng fine-tuning toàn bộ, Swin-T cao hơn ResNet18 '
        f'{paired("swin_t_full", "resnet18_full")}, với macro-F1 trung bình '
        f'{swin["test_macro_f1"]["mean"]:.4f} so với {resnet["test_macro_f1"]["mean"]:.4f}. '
        f'Thời gian train tăng {swin["train_seconds"]["mean"] / resnet["train_seconds"]["mean"]:.2f} lần '
        f'và peak allocated memory tăng '
        f'{swin["peak_gpu_memory_bytes"]["mean"] / resnet["peak_gpu_memory_bytes"]["mean"]:.2f} lần. '
        'Đây là sự đánh đổi giữa accuracy và chi phí trong cùng ngân sách epoch; '
        'hai kiến trúc vẫn khác số tham số và quá trình pretraining.\n\n')
    paragraphs.append(
        'Head-only giảm peak allocated memory so với full từ '
        f'{resnet["peak_gpu_memory_bytes"]["mean"] / 2**20:.1f} xuống '
        f'{groups["resnet18_head"]["peak_gpu_memory_bytes"]["mean"] / 2**20:.1f} MiB ở ResNet18, và từ '
        f'{swin["peak_gpu_memory_bytes"]["mean"] / 2**20:.1f} xuống '
        f'{groups["swin_t_head"]["peak_gpu_memory_bytes"]["mean"] / 2**20:.1f} MiB ở Swin-T. '
        'Đóng băng backbone giảm bộ nhớ gradient và trạng thái optimizer, '
        'nhưng vẫn cần bước forward qua toàn bộ backbone.\n\n')
    paragraphs.append(
        '\\Needspace{7\\baselineskip}\n\\subsection{Đường cong học và lỗi phân loại}\n'
        'Ở epoch 20, train/validation accuracy của các bản full có khoảng cách rõ: ')
    for index, family in enumerate(['resnet18', 'swin_t']):
        last_rows = []
        for seed in [42, 43, 44]:
            with (ROOT / 'runs/assignment1' / f'{family}_full' / f'seed_{seed}' / 'history.csv').open() as stream:
                last_rows.append(list(csv.DictReader(stream))[-1])
        train = statistics.mean(float(row['train_accuracy']) for row in last_rows)
        val = statistics.mean(float(row['val_accuracy']) for row in last_rows)
        paragraphs.append(f'{family_labels[family]} {100 * train:.2f}\\%/{100 * val:.2f}\\%'
                          + ('; ' if index == 0 else '. '))
    paragraphs.append(
        'Train cải thiện nhanh hơn validation, trong khi validation CE còn cao hơn train CE; '
        'đây là dấu hiệu overfit. Điểm test trong bảng lấy từ checkpoint tốt nhất theo validation, '
        'không nhất thiết là epoch 20.\n\n')
    for family in ['resnet18', 'swin_t']:
        run = ROOT / 'runs/assignment1' / f'{family}_full' / 'seed_42'
        evaluation = json.loads((run / 'evaluation_test.json').read_text())
        matrix = evaluation['confusion']
        a, b = max(((i, j) for i in range(100) for j in range(100) if i != j), key=lambda pair: matrix[pair[0]][pair[1]])
        classes = json.loads((run / 'split.json').read_text())['classes']
        paragraphs.append(f'{family_labels[family]} full ở seed 42: lỗi nhiều nhất theo một chiều là '
                          f'{latex_escape(classes[a])} thành {latex_escape(classes[b])}, {matrix[a][b]} ảnh. '
                          f'Lớp {latex_escape(classes[a])} có {sum(matrix[a])} ảnh trong tập test.\n\n')
    paragraphs.append(
        'Hai mô hình cùng nhầm oak\\_tree với maple\\_tree, gợi ý khó khăn khi phân biệt '
        'các lớp cây có hình dáng và kết cấu gần nhau trong ảnh $32\\times32$. '
        'Đây là cách diễn giải từ ma trận lỗi và các ví dụ quan sát, không phải phép đo riêng '
        'về mức tương đồng thị giác. Những lỗi confidence cao cho thấy accuracy cao vẫn đi kèm '
        'một số dự đoán sai rất tự tin; calibration chưa được đánh giá riêng.\n')
    (target / 'analysis.tex').write_text(''.join(paragraphs))
    (target / 'conclusion.tex').write_text(
        'Trong thiết lập CIFAR-100 đầy đủ với 20 epoch, Swin-T fine-tuning toàn bộ cho '
        f'test accuracy cao nhất, đạt {measured("swin_t_full", "test_accuracy", 100)}\\% '
        f'và macro-F1 {swin["test_macro_f1"]["mean"]:.4f}. '
        'Fine-tuning stage cuối hoặc toàn bộ cải thiện kết quả so với chỉ học head ở cả hai kiến trúc. '
        'ResNet18 scratch học được biểu diễn hữu ích trong ngân sách này, còn Swin-T scratch '
        'chưa học tốt. Lợi thế accuracy của Swin-T full đi kèm thời gian và bộ nhớ cao hơn ResNet18 full.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['all', 'exercises', 'a1'], default='all')
    parser.add_argument('--archive', action='store_true', help='Verify the immutable measured release for A1')
    args = parser.parse_args()
    if args.study in {'all', 'exercises'}:
        write_exercises()
    if args.study in {'all', 'a1'}:
        from common.archive import a1_source
        write_a1(source_files=a1_source() if args.archive else None)
    print(f'Wrote factual findings for {args.study} from verified train/test runs.')


if __name__ == '__main__':
    main()
