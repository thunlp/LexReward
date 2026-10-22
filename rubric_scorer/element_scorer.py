#!/usr/bin/env python3
"""Element rubric scorer using DeepSeek-V4-Flash LLM judge.

Evaluates Chinese legal responses across four dimensions: subjects (legal
parties), facts (case elements), statutes (legal grounds), and disposition
(conclusions). Each dimension is scored 0-10 with one decimal place. The
overall score is the arithmetic mean of the four dimensions.

This is a reference-free blind evaluator: it sees only the question and
candidate response, not the gold answer. It is appropriate for comparing
model-generated responses but should not be used when the gold answer is
one of the candidates being scored.

Usage:
    python rubric_scorer/element_scorer.py --file response.txt --json
    python rubric_scorer/element_scorer.py --jsonl candidates.jsonl

The MODELBEST_API_KEY environment variable must be set.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import aiohttp

DIMS = ('subjects', 'facts', 'statutes', 'disposition')

SYSTEM_PROMPT = '''你是严谨的法律要素评审专家。你会看到一道题和一个候选答案。
**没有参考答案**，请只依据题目要求判断候选答案的质量。
只评价以下四个 Element 维度，每项 0 到 10 分，允许一位小数：
subjects 主体认定：当事人/主体的身份、地位、关系是否认定准确；
facts 事实要素：与定性相关的关键事实要素是否准确、完整地识别；
statutes 法条依据：援引的法律规范、构成要件是否正确且切题；
disposition 处理结论：最终结论（选项、定性、责任分配、判决方向）是否正确、明确。

评分纪律（务必遵守）：
1. 严禁因为篇幅长、排版好、论证详尽而加分。题目只要求给出结论时，
   一个只给出正确结论、没有任何解释的简短答案，在 disposition 上应当拿满分。
2. 某个维度在这道题里不适用时（例如纯摘要题没有法条），该维度给 5 分，
   不要为了拉开差距而乱给分。
3. 请按真实质量独立打分。

只输出合法 JSON，不要输出 Markdown：
{"subjects":8.0,"facts":8.0,"statutes":8.0,"disposition":8.0}'''


def build_messages(question, answer):
    """Build the judge prompt messages."""
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'【题目】\n{question}\n\n【候选答案】\n{answer}'},
    ]


def extract_json(text):
    """Extract JSON from model response, handling markdown wrappers."""
    text = (text or '').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            raise ValueError(f'no json in response: {text[:200]}')
        return json.loads(m.group(0))


def parse_scores(raw_response):
    """Parse dimension scores from judge response."""
    obj = extract_json(raw_response)
    return {dim: float(obj[dim]) for dim in DIMS}


async def score_one(session, url, key, model, question, answer, retries=3):
    """Score a single response with retry logic."""
    messages = build_messages(question, answer)
    for attempt in range(retries):
        try:
            async with session.post(
                    url,
                    headers={'Authorization': f'Bearer {key}'},
                    json={'model': model, 'messages': messages, 'temperature': 0.0},
                    timeout=aiohttp.ClientTimeout(total=180)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(8)
                    continue
                if resp.status != 200:
                    raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:200]}')
                raw = (await resp.json())['choices'][0]['message']['content']
                scores = parse_scores(raw)
                overall = sum(scores.values()) / len(scores)
                return {'dimensions': scores, 'overall': overall}
        except (ValueError, RuntimeError, KeyError, asyncio.TimeoutError, aiohttp.ClientError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f'Failed after {retries} attempts: {exc}')
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError('Unexpected retry exhaustion')


async def main_async(args):
    key = os.environ.get('MODELBEST_API_KEY')
    if not key:
        raise SystemExit('MODELBEST_API_KEY environment variable not set')

    if args.file:
        text = Path(args.file).read_text(encoding='utf-8').strip()
        if not args.question:
            raise SystemExit('--question is required when using --file')

        conn = aiohttp.TCPConnector(limit=1, ssl=False)
        async with aiohttp.ClientSession(connector=conn, trust_env=True) as session:
            result = await score_one(session, args.url, key, args.model,
                                     args.question, text)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Element Rubric Scores:")
            for dim in DIMS:
                print(f"  {dim}: {result['dimensions'][dim]:.1f}")
            print(f"  overall: {result['overall']:.1f}")

    elif args.jsonl:
        items = []
        for line in Path(args.jsonl).open(encoding='utf-8'):
            if line.strip():
                items.append(json.loads(line))

        conn = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)
        async with aiohttp.ClientSession(connector=conn, trust_env=True) as session:
            sem = asyncio.Semaphore(args.concurrency)

            async def score_item(item):
                async with sem:
                    question = item.get('question') or item.get('instruction', '')
                    if item.get('input'):
                        question += '\n\n' + item['input']
                    answer = item.get('answer') or item.get('response', '')
                    result = await score_one(session, args.url, key, args.model,
                                             question, answer)
                    return {**item, 'element_scores': result}

            results = await asyncio.gather(*(score_item(it) for it in items))

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as out:
                for res in results:
                    out.write(json.dumps(res, ensure_ascii=False) + '\n')
            print(f"Wrote {len(results)} scored items to {args.output}")
        else:
            for res in results:
                print(json.dumps(res, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file', help='Score a single text file')
    p.add_argument('--question', help='Question text when using --file')
    p.add_argument('--jsonl', help='Score a JSONL file with question/answer pairs')
    p.add_argument('--output', help='Output path for scored JSONL')
    p.add_argument('--json', action='store_true',
                   help='Output JSON format (for --file mode)')
    p.add_argument('--url', default='https://llm-center.modelbest.co/v1/chat/completions',
                   help='API endpoint URL')
    p.add_argument('--model', default='deepseek-v4-flash',
                   help='Model name')
    p.add_argument('--concurrency', type=int, default=6,
                   help='Concurrent requests for JSONL mode')
    args = p.parse_args()

    if not args.file and not args.jsonl:
        p.error('Either --file or --jsonl is required')

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
