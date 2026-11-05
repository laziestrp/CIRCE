# SPDX-License-Identifier: Apache-2.0
"""Frozen, score/gold-blind final answer extraction; no semantic judge."""
import re
from decimal import Decimal, InvalidOperation

VERSION='final_answer_exact_v1'
DECIMAL=r'[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)'
NUMBER=r'(?<![\w.])'+DECIMAL+r'(?!\d|\.\d)'
OPERAND=r'(?:\\boxed\{\s*'+DECIMAL+r'\s*\}|'+DECIMAL+r')'
BINARY=re.compile(OPERAND+r'\s*(?:[+\-:–]|\b(?:to|through|plus|minus|times|x|divided by)\b)\s*'+OPERAND,re.I)
MARKER=re.compile(r'(?im)^[ \t]*(?:(?:so|thus|therefore)[,:]?\s*)?(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?::|is\b\s*:?)\s*')
CUE=re.compile(r'(?i)^(?:therefore|thus|hence|so|in conclusion)\b')
UNCERTAIN=re.compile(r'(?i)\b(?:if|intermediate|maybe|perhaps|either|roughly|approx\.?|approximately|about|cannot|uncertain|not)\b|\b(?:so far|at least|at most|up to|no (?:final )?answer|is wrong)\b|[~?]')
BOX=re.compile(r'^[\s$\\\[\]()]*\\boxed\{([^{}]+)\}[\s.$\\\[\]()]*$')

def clean(text):
    text=str(text).replace('\u00a0',' ').replace('\u2212','-').replace('\u202f',' ')
    # Strip paired bold wrappers, not exponent/multiplication operators.
    return '\n'.join(re.sub(r'(?<!\w)\*\*([^\n]*?)\*\*(?!\w)',r'\1',line).replace('`','')
        for line in text.splitlines() if not line.strip().startswith('```')).strip()

def extract(dataset,text):
    if dataset not in ('gsm8k','commonsenseqa'):raise ValueError('unsupported supplement dataset')
    text=clean(text); lines=[line.strip() for line in text.splitlines() if line.strip()]
    if not lines:return dict(value=None,reason='empty',source=None)
    markers=list(MARKER.finditer(text)); source='explicit_answer'
    if markers:segment=text[markers[-1].end():].strip()
    else:
        segment=lines[-1]; source='last_line'
        cue_removed=re.sub(r'(?i)^\s*(?:therefore|thus|hence|so|in conclusion)[,:]?\s*','',segment)
        box=BOX.fullmatch(cue_removed)
        if box:segment=box.group(1);source='last_line_boxed'
        elif not CUE.match(segment):
            bare=segment.strip(' \t$()[]\\')
            bare=re.sub(r'(?<=\d)\.$','',bare)
            pattern=r'[A-Ea-e]' if dataset=='commonsenseqa' else DECIMAL
            if re.fullmatch(pattern,bare) is None:return dict(value=None,reason='no_final_answer_marker',source=source)
            segment=bare;source='bare_last_line'
    if not segment:return dict(value=None,reason='empty_answer_block',source=source)
    if UNCERTAIN.search(segment) or re.search(r'(?im)^\s*(?:reasoning|analysis|working)\s*:',segment):
        return dict(value=None,reason='uncertain_or_nonfinal_answer',source=source)
    boxed=BOX.fullmatch(segment)
    if boxed:segment=boxed.group(1)
    if dataset=='commonsenseqa':
        segment=re.sub(r'(?i)^\s*(?:therefore|thus|hence|so|in conclusion)[,:]?\s*','',segment).strip()
        segment=re.sub(r'(?i)^\s*(?:(?:therefore|thus|hence|so)[,:]?\s*)?(?:the\s+)?(?:correct\s+)?(?:option\s+is\s*:?)','',segment).strip()
        leading=re.match(r'(?i)^\s*(?:option\s*)?(?:\(([A-E])\)|\[([A-E])\]|([A-E]))(?=$|[\s.:,;-])',segment)
        if not leading:return dict(value=None,reason='no_leading_option',source=source)
        value=next(v for v in leading.groups() if v is not None).upper(); tail=segment[leading.end():]
        alternatives=re.search(r'(?i)(?:[/,;]|\b(?:or|and|vs|versus|alternatively|choose|rather than)\b)\s*(?:else\s+|option\s*)?[\[(]?[A-E]\b',tail)
        other_letters=[v for v in re.findall(r'\b[A-E]\b',tail) if v!=value]
        if '?' in segment or alternatives or other_letters:
            return dict(value=None,reason='ambiguous_option',source=source)
        return dict(value=value,reason='extracted',source=source)
    if BINARY.search(segment) or re.search(r'[/<>≤≥?*×÷^]|\b(?:or|between|approximately|about)\b',segment,re.I):
        return dict(value=None,reason='ambiguous_numeric_expression',source=source)
    if any(command not in {'boxed','text','mathrm','left','right'} for command in re.findall(r'\\([A-Za-z]+)',segment)):
        return dict(value=None,reason='unsupported_numeric_expression',source=source)
    if re.search(r'\d[eE][+-]?\d|[+\-]\s*[+\-]\s*\d',segment):
        return dict(value=None,reason='malformed_or_unsupported_number',source=source)
    for raw in re.findall(r'[+-]?(?:\d[\d,.]*|\.\d[\d,.]*)',segment):
        raw=raw.rstrip(',')
        if raw.endswith('.') and not raw.endswith('..'):raw=raw[:-1]
        if re.fullmatch(DECIMAL,raw) is None:
            return dict(value=None,reason='malformed_number',source=source)
    tokens=re.findall(NUMBER,segment)
    try:values={Decimal(token.replace(',','')) for token in tokens}
    except InvalidOperation:return dict(value=None,reason='malformed_number',source=source)
    if len(values)!=1:return dict(value=None,reason='no_number' if not values else 'multiple_numeric_values',source=source)
    value=next(iter(values))
    if not value.is_finite():return dict(value=None,reason='nonfinite_number',source=source)
    return dict(value=str(value),reason='extracted',source=source)

def label(dataset,text,gold):
    found=extract(dataset,text)
    if found['value'] is None:correct=False
    elif dataset=='commonsenseqa':correct=found['value']==str(gold).strip().upper()
    else:
        try:correct=Decimal(found['value'])==Decimal(str(gold).replace(',',''))
        except InvalidOperation:correct=False
    return dict(correct=bool(correct),**found)
