from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.units import mm
import textwrap

in_path = "docs/full_pipeline.md"
out_path = "docs/full_pipeline.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Heading1', parent=styles['Heading1'], spaceAfter=6))
styles.add(ParagraphStyle(name='Heading2', parent=styles['Heading2'], spaceAfter=4))
styles.add(ParagraphStyle(name='Code', parent=styles['Code'], fontName='Courier'))

flowables = []

with open(in_path, 'r') as f:
    lines = f.readlines()

in_code = False
code_buf = []
para_buf = []

def flush_para():
    global para_buf
    if para_buf:
        text = ' '.join(p.strip() for p in para_buf)
        flowables.append(Paragraph(text, styles['Normal']))
        flowables.append(Spacer(1,4*mm))
        para_buf = []

for line in lines:
    s = line.rstrip('\n')
    if s.startswith('```'):
        if not in_code:
            in_code = True
            code_buf = []
        else:
            # end code
            in_code = False
            flush_para()
            code_text = '\n'.join(code_buf)
            flowables.append(Preformatted(code_text, styles['Code']))
            flowables.append(Spacer(1,4*mm))
            code_buf = []
        continue

    if in_code:
        code_buf.append(s)
        continue

    if s.startswith('# '):
        flush_para()
        flowables.append(Paragraph(s[2:].strip(), styles['Heading1']))
        flowables.append(Spacer(1,2*mm))
        continue
    if s.startswith('## '):
        flush_para()
        flowables.append(Paragraph(s[3:].strip(), styles['Heading2']))
        flowables.append(Spacer(1,1*mm))
        continue
    if s.startswith('- '):
        para_buf.append('• ' + s[2:].strip())
        continue
    if s.strip() == '':
        flush_para()
        continue
    if s.startswith('```'):
        continue
    # normal text: accumulate until blank line
    para_buf.append(s)

flush_para()

# build PDF
pdf = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
pdf.build(flowables)

print('WROTE', out_path)
