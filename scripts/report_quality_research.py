"""Report valid single-DOE improvements separately from noncompliant controls."""
import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

plt.rcParams['pdf.fonttype']=42


def read(path):
    return json.loads(Path(path).read_text())


def generate(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    base=read('deliverables/texture_finetuned/benchmark.json')['summary']['neural']
    benchmark=read('artifacts/quality_research/qat_test_with_refinement.json')
    training=read('artifacts/perceptual_qat_v1/training.json')
    user_base=read('deliverables/texture_finetuned/result/metrics.json')
    user={n:read(f'artifacts/quality_research/qat_user_{n}/metrics.json') for n in (0,600,3000)}
    probes={p.stem:read(p) for p in sorted(Path('artifacts/quality_research').glob('validation_*.json'))}
    rows=[]
    def add(split,name,pearson,ssim,seconds=None,compliant=True):
        rows.append({'split':split,'method':name,'pearson':pearson,'ssim':ssim,'seconds_per_quartet':seconds,
                     'one_shared_doe':compliant,'both_at_least_0_8':bool(compliant and pearson>=0.8 and ssim>=0.8)})
    add('test_32','original neural',base['mean_pearson'],base['mean_ssim'],base['seconds_per_quartet_batched'])
    for key in ('neural','gs_80','neural_refined'):
        value=benchmark['summary'][key]
        add('test_32',{'neural':'QAT neural candidate','gs_80':'GS 80','neural_refined':'QAT neural + 600 Adam steps'}[key],
            value['mean_pearson'],value['mean_ssim'],value['seconds_per_quartet_batched'])
    add('user_quartet','original neural',user_base['mean_pearson'],user_base['mean_ssim'],user_base['metadata']['network_forward_seconds'])
    for n,r in user.items():
        seconds=r['metadata']['network_forward_seconds']+r['metadata'].get('refinement',{}).get('seconds',0)
        add('user_quartet',f'QAT neural + {n} Adam steps',r['mean_pearson'],r['mean_ssim'],seconds)
    for name,p in probes.items():
        last=p['history'][-1]
        compliant=not p['config'].get('independent_planes',False)
        add('validation_first_4',name,last['mean_pearson'],last['mean_ssim'],p['seconds']/p['config']['count'],compliant)
    with (output/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    summary={'goal':{'pearson':0.8,'ssim':0.8,'one_shared_doe':True},'goal_achieved':False,
             'checkpoint_selection':'QAT selected by validation perceptual loss; test used only for final evaluation',
             'production_weights_replaced':False,'measurements':rows,'qat_training':training,
             'limitations':['The four-independent-DOE control is not compliant with the original task.',
                            'Failure of these experiments is not a mathematical impossibility proof.',
                            '17 tests passed; GPU/Kaggle execution was not available.']}
    assert not any(r['both_at_least_0_8'] for r in rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    test_rows=[r for r in rows if r['split']=='test_32']
    for ax,key in zip(axes,('pearson','ssim')):
        bars=ax.bar(np.arange(len(test_rows)),[r[key] for r in test_rows],color=['#7c8792','#a5b5c3','#d8984c','#2f80b8'])
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.axhline(.8,color='#b83030',ls='--',label='Goal 0.8')
        ax.set_xticks(np.arange(len(test_rows)),['Original NN','QAT NN','GS 80','NN + Adam 600'],rotation=18)
        ax.set(ylim=(0,1),ylabel=key.upper(),title='Same test split: 32 quartets')
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'test_quality.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    selected=['validation_perceptual_600','validation_neural_3000','validation_independent_planes_600']
    labels=['1 DOE, 600 steps','1 DOE, 3000 steps','4 independent DOEs\nNOT COMPLIANT']
    for ax,key in zip(axes,('mean_pearson','mean_ssim')):
        values=[probes[n]['history'][-1][key] for n in selected]
        bars=ax.bar(np.arange(3),values,color=['#668fba','#2f80b8','#cf8484'])
        bars[-1].set_hatch('//')
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.axhline(.8,color='#b83030',ls='--')
        ax.set_xticks(np.arange(3),labels,fontsize=8)
        ax.set(ylim=(0,1.05),title=key.replace('mean_','').upper(),ylabel='Validation first 4 quartets')
    fig.savefig(output/'constraint_control.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(12,12),constrained_layout=True)
    for i in range(4):
        paths=[Path(f'artifacts/quality_research/qat_user_0/target_{i+1}.png')]+[
            Path(f'artifacts/quality_research/qat_user_{n}/reconstruction_{i+1}.png') for n in (0,600,3000)]
        for j,(ax,path) in enumerate(zip(axes[i],paths)):
            with Image.open(path) as image: ax.imshow(np.array(image),cmap='gray',vmin=0,vmax=255)
            ax.axis('off')
            if j==0: ax.set_title(f'Target, z={user[0]["planes"][i]["distance_mm"]:.1f} mm',fontsize=9)
            else:
                n=(0,600,3000)[j-1];r=user[n]['planes'][i]
                ax.set_title(f'{n} steps: r={r["pearson"]:.3f}\nSSIM={r["ssim"]:.3f}',fontsize=9)
    fig.savefig(output/'user_comparison.png',dpi=150);plt.close(fig)
    table='| Набор | Метод | Pearson | SSIM | Один общий ДОЭ |\n|---|---|---:|---:|---|\n'
    for r in rows:
        table+=f"| {r['split']} | {r['method']} | {r['pearson']:.5f} | {r['ssim']:.5f} | {'да' if r['one_shared_doe'] else '**нет — контроль**'} |\n"
    markdown=f'''# Исследование цели 0.8 по Pearson и SSIM

**При одном общем ДОЭ цель пока не достигнута.** Это исследование и улучшение
качества, а не заявление о выполнении целевого порога.

## Фактические результаты

{table}

![Тест](test_quality.png)

![Контроль ограничения](constraint_control.png)

![Пользовательская четвёрка](user_comparison.png)

## Что изменено

- Добавлен дифференцируемый SSIM, проверенный против skimage и конечными разностями.
- Добавлено обучение с прямым perceptual loss и straight-through квантованием в 8 бит.
  12 эпох QAT дали лишь малое улучшение validation loss; на test обе метрики чистой
  сети немного снизились. Поэтому прежние production-веса не заменены.
- Проверены Adam, LBFGS, разные chirp-инициализации, 600 и 3000 шагов фазы.
- Добавлен опциональный `--refine-steps`: сеть плюс итеративный решатель одной общей
  фазы. Это работает медленнее одного forward pass; число шагов и время явно записаны.
- В контрольном опыте каждому изображению разрешён свой отдельный ДОЭ.
  Высокие 0.99/0.95 относятся к этой другой постановке, не к выполнению задания.

## Корректность сравнения

Физика Френеля, λ, расстояния и размер 128×128 не менялись. Сохраняется амплитуда,
не интенсивность. Все финальные метрики считают фактические реконструкции из
сохранённого 8-битного PNG ДОЭ; нет фильтрации, подгонки gain по target или ROI.
Настройки выбирались по validation, после чего выполнена оценка на тех же 32 test
четвёрках. Для исследовательского подбора использованы первые 4 validation-четвёрки,
поэтому эти малые оценки не являются новым независимым test.
Результаты с разными split нельзя сравнивать как показатели одной выборки.
Итеративный solver использует желаемые изображения как входы обратной задачи;
его качество не выдаётся за качество чистой нейросети. Времена — текущий CPU,
batched throughput, без статистических гарантий. Опыты не доказывают математической
невозможности 0.8 с одним ДОЭ.

## Следующие направления

1. Более мощный обучаемый инициализатор волнового фронта (U-Net/HoloNet),
   пространственные маски амплитудной свободы и больше обучаемых optical stages.
   Это перспективные изменения архитектуры, **не проверенное обещание 0.8**.
2. Teacher/student дистилляция итеративного решателя. Сначала учитель должен
   показать нужное качество: текущие результаты сами по себе 0.8 не обеспечивают.
3. При разрешённом изменении задания — отдельные ДОЭ/экспозиции или больше
   управляемых пикселей. Четыре независимых ДОЭ показали высокий результат в
   контрольном опыте, но исходное условие одного ДОЭ при этом нарушается.

Источники: [Neural Holography](https://github.com/computational-imaging/neural-holography)
и [Tensor Holography](https://github.com/liangs111/tensor_holography). Прочитаны
официальные README и опубликованный алгоритм SGD. Их оценки не переносились на
нашу постановку, код/веса не копировались. Дополнительные источники и исходные
истории находятся в `docs/quality-research.md` и каталоге probes архива.

Проверки: **17 тестов pytest**. Полные CPU-обучение и benchmark выполнены.
Kaggle/GPU и эксперимент на физическом ПВМС не выполнялись.
'''
    (output/'report.md').write_text(markdown)
    refined=benchmark['summary']['neural_refined']
    with PdfPages(output/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69))
        fig.text(.07,.94,'Исследование качества ДОЭ: цель 0.8',fontsize=17,weight='bold')
        paragraphs=[
            'Цель: Pearson ≥0.8 и SSIM ≥0.8 одновременно, один общий фазовый ДОЭ 128×128 для четырёх плоскостей. В этом исследовании цель пока не достигнута.',
            f"На тех же 32 test-четвёрках исходная сеть: {base['mean_pearson']:.5f}/{base['mean_ssim']:.5f}; сеть + 600 шагов фазы: {refined['mean_pearson']:.5f}/{refined['mean_ssim']:.5f} (Pearson/SSIM).",
            f"Пользовательская четвёрка: {user_base['mean_pearson']:.5f}/{user_base['mean_ssim']:.5f} → {user[3000]['mean_pearson']:.5f}/{user[3000]['mean_ssim']:.5f} с 3000 шагами. Это итеративный решатель после сети, не один forward pass.",
            f"Время сети + 600 шагов: {refined['seconds_per_quartet_batched']:.2f} с/четвёрку в пакетном CPU-тесте. Увеличение качества оплачивается временем расчёта.",
            'QAT/perceptual fine-tuning: 12 эпох. На test чистая сеть немного хуже прежней, поэтому production-веса сохранены. Кандидат опубликован отдельно.',
            'Контроль: четыре независимых ДОЭ на первые 4 validation-четвёрки дали около 0.990/0.951. Это другая постановка: исходному условию одного общего ДОЭ она не соответствует.',
            'Физика, входы и метрики сохранены. Оценка после настоящего 8-битного PNG ДОЭ, без фильтрации, ROI и gain по target. Отрицательный результат не является доказательством невозможности 0.8.',
            'Дальше: обучаемый wavefront-инициализатор U-Net/HoloNet и больше optical stages; дистилляция качественного учителя. Успех до 0.8 не подтверждён. Дополнительные ДОЭ требуют изменения задания.',
            'Источники: Neural Holography (Peng et al.) и Tensor Holography (Shi et al.). Прочитаны опубликованные материалы; код и веса не копировались. 17 тестов пройдены. GPU/Kaggle и физический ПВМС не проверены.'
        ]
        y=.88
        for paragraph in paragraphs:
            lines=textwrap.wrap(paragraph,91)
            fig.text(.07,y,'\n'.join(lines),fontsize=10,va='top',linespacing=1.4)
            y-=len(lines)*.019+.023
        pdf.savefig(fig);plt.close(fig)
        for name,title,figsize in [('test_quality.png','Одинаковый test: 32 четвёрки',(10,6)),
                                   ('constraint_control.png','Контроль: одинаковые 4 validation-четвёрки',(10,6)),
                                   ('user_comparison.png','Пользовательские изображения: исходы и реконструкции',(8.27,11.69))]:
            fig,ax=plt.subplots(figsize=figsize)
            with Image.open(output/name) as image: ax.imshow(np.array(image))
            ax.axis('off');ax.set_title(title,fontsize=12)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print(json.dumps({'report':str(output),'goal_achieved':False,'test_neural_refined':refined},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    generate(p.parse_args().output)
