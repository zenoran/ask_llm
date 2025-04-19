import pytest
from datetime import datetime, timezone
import pytz

import ask_llm.model_manager as mm

def test_format_model_timestamp_str_naive_and_tz():
    tz = pytz.timezone('UTC')
    # naive datetime
    dt = datetime(2020,1,1,12,0,0)
    s = mm.ModelManager._format_model_timestamp_str(None, 'openai', {'created': dt}, tz)
    assert '2020-01-01' in s
    # with tzinfo
    dt2 = datetime(2020,1,1,12,0,0, tzinfo=timezone.utc)
    s2 = mm.ModelManager._format_model_timestamp_str(None, 'openai', {'created': dt2}, tz)
    assert '2020-01-01' in s2
    # None dt returns None
    assert mm.ModelManager._format_model_timestamp_str(None, 'openai', {}, tz) is None

def test_generate_alias():
    gen = mm.ModelManager._generate_alias
    # simulate collisions
    existing = {'mod', 'mod1'}
    alias = gen(None, None, {'id':'mod'}, existing_aliases=existing)
    # Should skip mod and mod1, return mod2
    # But signature: (self, provider_type, model_info, existing_aliases)
    # So call with dummy self and info
    alias = gen(None, 'type', {'id':'mod'}, existing_aliases=existing)
    assert alias == 'mod2'

@pytest.mark.parametrize('model_type, info, expected', [
    ('gguf', {'repo_id':'r','chat_format':'f'}, 'Repo: r, Format: f'),
    ('gguf', {'repo_id':'r','description':'Repo: r abc'}, 'Repo: r abc'),
    ('huggingface', {'model_id':'mid'}, 'ID: mid'),
    ('ollama', {'model_id':'mid'}, 'ID: mid'),
    ('openai', {'model_id':'mid','description':'desc'}, 'desc'),
    ('openai', {'model_id':'mid'}, 'ID: mid'),
])
def test_format_model_details(model_type, info, expected):
    out = mm.ModelManager._format_model_details(None, model_type, info)
    assert expected in out

def test_get_dependency_note():
    assert 'ask-llm[huggingface]' in mm.ModelManager._get_dependency_note('huggingface')
    assert 'llama-cpp-python' in mm.ModelManager._get_dependency_note('gguf')
    assert mm.ModelManager._get_dependency_note('openai') == ''

def test_prompt_for_new_models(monkeypatch):
    new_models = [{'id':'m1'},{'id':'m2'}]
    choices = {'1':new_models[0],'2':new_models[1]}
    # Test skip
    monkeypatch.setattr(mm.Prompt, 'ask', lambda prompt, default='': '')
    sel = mm.ModelManager._prompt_for_new_models(None, new_models, choices)
    assert sel == []
    # Test select both
    # For openai sep=','; simulate
    monkeypatch.setattr(mm.Prompt, 'ask', lambda prompt, default='': '1,2')
    sel2 = mm.ModelManager._prompt_for_new_models(None, new_models, choices)
    assert sel2 == new_models

def test_prompt_for_match(monkeypatch):
    matches = ['a','b','c']
    # patch Prompt.ask
    monkeypatch.setattr(mm.Prompt, 'ask', lambda prompt, choices, default='': '3')
    res = mm.ModelManager._prompt_for_match(None, 'req', matches)
    assert res == 'c'