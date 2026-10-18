import numpy as np
import pandas as pd
import pytest

from tools.analyze_original_review import paired,cluster_statistics,operational_metrics,holm


def test_statistics_use_clusters_not_correlated_windows_or_model_seeds():
    frame = pd.DataFrame({'dataset':['a']*4+['b']*4,'scenario':['s']*8,
                          'difference':[.1,.2,.1,.2,-.1,-.2,-.1,-.2]})
    result = cluster_statistics(frame,['dataset','scenario'])
    assert result['clusters']==2 and result['paired_cells']==8
    assert abs(result['mean_difference'])<1e-12


def test_incomplete_and_duplicate_pairs_raise():
    frame = pd.DataFrame([dict(dataset='a',scenario='s',stream_seed=1,model_seed=1,
        condition='native',mean_macro_f1=.8),dict(dataset='b',scenario='s',stream_seed=1,
        model_seed=1,condition='random',mean_macro_f1=.9)])
    with pytest.raises(ValueError,match='incomplete'):
        paired(frame,'native','random')
    with pytest.raises(ValueError,match='duplicate'):
        paired(pd.concat([frame,frame]),'native','random')


def test_operational_error_tradeoffs_do_not_confuse_mean_with_cumulative():
    frame = pd.DataFrame(dict(warmup=[False,False],attack_tp=[2,0],attack_fn=[0,0],
        attack_fp=[1,0],attack_tn=[0,99]))
    result = operational_metrics(frame)
    assert result['mean_window_benign_fpr']==.5
    assert result['cumulative_benign_fpr']==.01
    assert result['cumulative_attack_recall']==1.


def test_holm_correction_and_single_cluster_limitations():
    records = [{'test':{'p_two_sided':p}} for p in (.03,.001,.02)]
    holm(records,'test')
    assert np.allclose([r['test']['holm_adjusted_p'] for r in records],[.04,.003,.04])
    result = cluster_statistics(pd.DataFrame({'dataset':['a'],'difference':[.1]}),['dataset'])
    assert result['p_two_sided'] is None
