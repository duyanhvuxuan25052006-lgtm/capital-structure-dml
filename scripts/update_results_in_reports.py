"""
TỰ ĐỘNG CẬP NHẬT KẾT QUẢ VÀO CÁC BÁO CÁO (MD & TEX)
===================================================
Script này đọc kết quả hồi quy mới nhất từ:
  - results/baseline_results.csv
  - results/dml_results_expanded.csv
và tự động chèn các bảng biểu được định dạng đẹp mắt vào các báo cáo Markdown & LaTeX.
===================================================
"""
import os, sys
import pandas as pd
import numpy as np

workspace_md = r"D:\draft 2\paper\Capital_Structure_DML_Report.md"
workspace_tex = r"D:\draft 2\paper\Capital_Structure_DML_Report.tex"
brain_md = r"C:\Users\duyan\.gemini\antigravity\brain\c5f57ce7-90d9-4bc1-9cb7-7ef55489e851\Capital_Structure_DML_Report.md"
brain_tex = r"C:\Users\duyan\.gemini\antigravity\brain\c5f57ce7-90d9-4bc1-9cb7-7ef55489e851\Capital_Structure_DML_Report.tex"

baseline_csv = r"D:\draft 2\results\baseline_results.csv"
dml_csv = r"D:\draft 2\results\dml_results_expanded.csv"

def format_pval(p):
    if p < 0.001:
        return "< 0,001"
    return f"{p:.4f}"

def format_sig(val, p):
    stars = ""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    return f"{val:+.4f}{stars}"

def generate_baseline_md(df):
    md = "| Biến can thiệp | Mô hình | Hệ số $\\theta$ | Sai số chuẩn | p-value | Khoảng tin cậy 95% | $R^2$ |\n"
    md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for _, r in df.iterrows():
        sig_theta = format_sig(r['theta'], r['pval'])
        p_val_str = format_pval(r['pval'])
        md += f"| **{r['Treatment']}** | {r['Model']} | {sig_theta} | {r['se']:.4f} | {p_val_str} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['r2']*100:.1f}% |\n"
    return md

def generate_baseline_tex(df):
    tex = "\\begin{tabular}{llccccc}\n\\toprule\nBiến D & Mô hình & Hệ số $\\theta$ & Sai số chuẩn & p-value & Khoảng tin cậy 95\\% & $R^2$ \\\\\n\\midrule\n"
    current_d = ""
    for _, r in df.iterrows():
        sig_theta = format_sig(r['theta'], r['pval']).replace("+", "$+$").replace("-", "$-$")
        p_val_str = format_pval(r['pval']).replace("<", "$<$")
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]".replace("-", "$-$")
        
        if current_d != r['Treatment'] and current_d != "":
            tex += "\\midrule\n"
        current_d = r['Treatment']
        
        tex += f"\\textbf{{{r['Treatment']}}} & {r['Model']} & {sig_theta} & {r['se']:.4f} & {p_val_str} & {ci_str} & {r['r2']*100:.1f}\\% \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    return tex

def generate_dml_md(df):
    md = "| Biến D | Đặc tả mẫu | Thuật toán | Hệ số $\\theta$ | Sai số chuẩn | p-value | RV (%) | RVa (%) |\n"
    md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for _, r in df.iterrows():
        # Handle cases where some fields might be NaN (like Lasso in Quadratic)
        t_val = r['RF_theta'] if not pd.isna(r['RF_theta']) else r.get('Lasso_theta', np.nan)
        if pd.isna(t_val):
            continue
            
        spec = r['Model_Specification']
        t_name = r['Treatment']
        y_name = r['Y_variable']
        
        # We can report RF as the primary model
        sig_theta = format_sig(r['RF_theta'], r['RF_pval']) if not pd.isna(r['RF_theta']) else "-"
        se_val = f"{r['RF_se']:.4f}" if not pd.isna(r['RF_se']) else "-"
        p_val_str = format_pval(r['RF_pval']) if not pd.isna(r['RF_pval']) else "-"
        rv_str = f"{r['RF_RV']:.2f}" if ('RF_RV' in r and not pd.isna(r['RF_RV'])) else "---"
        rva_str = f"{r['RF_RVa']:.2f}" if ('RF_RVa' in r and not pd.isna(r['RF_RVa'])) else "---"
        
        md += f"| **{t_name}** | {spec} | Random Forest | {sig_theta} | {se_val} | {p_val_str} | {rv_str} | {rva_str} |\n"
    return md

def generate_dml_tex(df):
    tex = "\\begin{tabular}{lllccccc}\n\\toprule\nBiến D & Mẫu nghiên cứu & Thuật toán & Hệ số $\\theta$ & SE & p-value & RV (\\%) & RVa (\\%) \\\\\n\\midrule\n"
    current_d = ""
    for _, r in df.iterrows():
        if pd.isna(r['RF_theta']):
            continue
        sig_theta = format_sig(r['RF_theta'], r['RF_pval']).replace("+", "$+$").replace("-", "$-$")
        p_val_str = format_pval(r['RF_pval']).replace("<", "$<$")
        rv_str = f"{r['RF_RV']:.2f}" if ('RF_RV' in r and not pd.isna(r['RF_RV'])) else "---"
        rva_str = f"{r['RF_RVa']:.2f}" if ('RF_RVa' in r and not pd.isna(r['RF_RVa'])) else "---"
        
        if current_d != r['Treatment'] and current_d != "":
            tex += "\\midrule\n"
        current_d = r['Treatment']
        
        spec = r['Model_Specification'].replace("&", "\\&").replace("%", "\\%")
        tex += f"\\textbf{{{r['Treatment']}}} & {spec} & Random Forest & {sig_theta} & {r['RF_se']:.4f} & {p_val_str} & {rv_str} & {rva_str} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    return tex

def update_report_text(report_path, table_type, new_table_content, file_format='md'):
    """Thay thế bảng cũ bằng bảng mới dựa trên cấu trúc file."""
    if not os.path.exists(report_path):
        print(f"Warning: {report_path} not found.")
        return
        
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    skip = False
    
    start_marker = f"<!-- BEGIN {table_type.upper()} TABLE -->" if file_format == 'md' else f"% BEGIN {table_type.upper()} TABLE"
    end_marker = f"<!-- END {table_type.upper()} TABLE -->" if file_format == 'md' else f"% END {table_type.upper()} TABLE"
    
    has_markers = any(start_marker in l for l in lines)
    
    if not has_markers:
        # Nếu chưa có marker, ta sẽ chèn động bằng cách tìm bảng cũ (fallback đơn giản)
        print(f"  Note: Khử chèn động cho {report_path} (chưa cấu hình marker, sẽ tự chèn marker).")
        # Thay thế trực tiếp trong text
        content = "".join(lines)
        if file_format == 'md':
            if table_type == 'baseline':
                # Tìm vị trí bảng baseline cũ
                start_idx = content.find("| Biến can thiệp | Mô hình |")
                if start_idx != -1:
                    # tìm dòng trống tiếp theo kết thúc bảng
                    end_idx = content.find("\n\n", start_idx)
                    content = content[:start_idx] + start_marker + "\n" + new_table_content + end_marker + content[end_idx:]
            elif table_type == 'dml':
                start_idx = content.find("| Biến D | Đặc tả mẫu |")
                if start_idx != -1:
                    end_idx = content.find("\n\n", start_idx)
                    content = content[:start_idx] + start_marker + "\n" + new_table_content + end_marker + content[end_idx:]
        else: # tex
            if table_type == 'baseline':
                start_idx = content.find("\\begin{table}")
                # tìm tabular đầu tiên
                tab_start = content.find("\\begin{tabular}", start_idx)
                tab_end = content.find("\\end{tabular}", tab_start) + len("\\end{tabular}")
                content = content[:tab_start] + start_marker + "\n" + new_table_content + end_marker + content[tab_end:]
            elif table_type == 'dml':
                # tìm table thứ 2
                start_idx = content.find("\\begin{table}", content.find("\\begin{table}") + 10)
                tab_start = content.find("\\begin{tabular}", start_idx)
                tab_end = content.find("\\end{tabular}", tab_start) + len("\\end{tabular}")
                content = content[:tab_start] + start_marker + "\n" + new_table_content + end_marker + content[tab_end:]
                
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return

    # Nếu đã có marker
    for line in lines:
        if start_marker in line:
            new_lines.append(line)
            new_lines.append(new_table_content + "\n")
            skip = True
        elif end_marker in line:
            new_lines.append(line)
            skip = False
        elif not skip:
            new_lines.append(line)
            
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    print("=" * 70)
    print("  TỰ ĐỘNG CẬP NHẬT KẾT QUẢ MỚI VÀO BÁO CÁO")
    print("=" * 70)
    
    if not os.path.exists(baseline_csv) or not os.path.exists(dml_csv):
        print("ERROR: Không tìm thấy tệp kết quả CSV! Vui lòng chạy run_all_analyses.py trước.")
        sys.exit(1)
        
    df_base = pd.read_csv(baseline_csv)
    df_dml = pd.read_csv(dml_csv)
    
    # 1. Tạo các bảng định dạng mới
    base_md = generate_baseline_md(df_base)
    base_tex = generate_baseline_tex(df_base)
    dml_md = generate_dml_md(df_dml)
    dml_tex = generate_dml_tex(df_dml)
    
    # 2. Cập nhật các tệp báo cáo
    for path, fmt in [(workspace_md, 'md'), (brain_md, 'md')]:
        print(f"Cập nhật Markdown: {path}")
        update_report_text(path, 'baseline', base_md, fmt)
        update_report_text(path, 'dml', dml_md, fmt)
        
    for path, fmt in [(workspace_tex, 'tex'), (brain_tex, 'tex')]:
        print(f"Cập nhật LaTeX: {path}")
        update_report_text(path, 'baseline', base_tex, fmt)
        update_report_text(path, 'dml', dml_tex, fmt)
        
    print("\n>>> ĐÃ CẬP NHẬT THÀNH CÔNG TOÀN BỘ SỐ LIỆU VÀO CÁC BÁO CÁO! <<<")

if __name__ == "__main__":
    main()
