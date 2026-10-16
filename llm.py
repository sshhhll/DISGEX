"""Adapt model requests and decode task responses.

Model branches normalize provider names, sampling options and output formats.
API wrappers send requests through the transport supplied by the caller.
Response decoders check output structure, while completion status and pending
requests keep interrupted calls separate from valid extraction results."""
import copy
import hashlib
import json
import re
from types import SimpleNamespace


# Translate logical model settings into provider request fields and structured output schemas.
def _crf_normalizer_gpt4o(CFG):
    def normalize(self, request):
        assert request.get("model") in (CFG["model"], CFG["model_alias"]), "Unexpected logical model"
        value = copy.deepcopy(request)
        value["model"] = CFG["wire_model"]
        value["temperature"] = CFG["temperature"]
        value.pop("enable_thinking", None)
        value.pop("thinking_budget", None)
        value.pop("cache", None)
        value.pop("rollout_id", None)
        maximum = value.pop("max_completion_tokens", None)
        if maximum is not None:
            value["max_tokens"] = maximum
        if value.get("max_tokens") is None:
            value["max_tokens"] = CFG["reader_max_tokens"]
        value.pop("chat_template_kwargs", None)
        value = {k: v for k, v in value.items() if v is not None}
        fmt = value.get("response_format")
        if isinstance(fmt, type) and hasattr(fmt, "model_json_schema"):
            value["response_format"] = {"type": "json_schema", "json_schema": {
                "name": fmt.__name__, "strict": True, "schema": fmt.model_json_schema()}}
        assert value["model"] == CFG["wire_model"] == "gpt-4o-2024-11-20"
        assert value["temperature"] == 0 and value["max_tokens"] <= 16384
        assert all(m["role"] in ("system", "user", "assistant") for m in value["messages"])
        return value
    return normalize


# Disable model thinking in the chat template while preserving the requested response structure.
def _crf_normalizer_qwen3_8b(CFG):
    def normalize(self, request):
        assert request.get("model") in (CFG["model"], CFG["model_alias"]), "Unexpected logical model"
        value = copy.deepcopy(request)
        value["model"] = CFG["model"]
        value["temperature"] = CFG["temperature"]
        value.pop("enable_thinking", None)
        value.pop("thinking_budget", None)
        value.pop("cache", None)
        value.pop("rollout_id", None)
        maximum = value.pop("max_completion_tokens", None)
        if maximum is not None:
            value["max_tokens"] = maximum
        if value.get("max_tokens") is None:
            value["max_tokens"] = CFG["reader_max_tokens"]
        value["chat_template_kwargs"] = {"enable_thinking": False}
        value = {k: v for k, v in value.items() if v is not None}
        fmt = value.get("response_format")
        if isinstance(fmt, type) and hasattr(fmt, "model_json_schema"):
            value["response_format"] = {"type": "json_schema", "json_schema": {
                "name": fmt.__name__, "strict": True, "schema": fmt.model_json_schema()}}
        assert value["model"] == "Qwen/Qwen3-8B"
        assert value["temperature"] == 0 and value["max_tokens"] <= 16384
        assert all(m["role"] in ("system", "user", "assistant") for m in value["messages"])
        return value
    return normalize

# Keep the requested Qwen sampling settings after shared request normalization.
def crf_api(transport, branch):
    """Normalize CRF requests before calling the supplied transport."""
    base=(_crf_normalizer_qwen3_8b if branch.profile == 'qwen3_8b' else _crf_normalizer_gpt4o)(branch.assets['config'])
    def call(request,purpose):
        value=base(None,request)
        if branch.profile == 'qwen3_8b' and request.get('temperature')==0.7:
            value.update(temperature=0.7,top_p=0.8,top_k=20,seed=42)
        return transport.call(value,purpose)
    return SimpleNamespace(call=call)


def _simord_order_parser():
    from dataclasses import dataclass
    from typing import Dict, Any

    @dataclass
    class Order:
        """Generic Order class"""
        description: str
        order_type: str
        reason: str
        provenance: list[str] = None

        def to_dict(self) -> Dict[str, Any]:
            """Convert the Order object to a dictionary."""
            return self.__dict__.copy()

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Order':
            """Create an Order object from a dictionary."""
            valid = {k: data.get(k, None) for k in cls.__annotations__.keys()}
            return cls(**valid)

    DEFAULT_EXECUTION_SETTINGS = {
        "max_tokens": 4320,
        "temperature": 0.0,
        "top_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "n": 1,
    }

    # Decode the JSON order array and retain the task field names for each order.
    def get_orders_from_model_response(response):
        raw_response = response.choices[0].message.content
        raw_response = raw_response.strip()
        raw_response = raw_response.replace("```json", "")
        raw_response = raw_response.replace("```", "")


        try:
            json_message = json.loads(raw_response) 
        except json.JSONDecodeError:


            return None, True


        orders_parsed = []
        for order in json_message:
            try:
                order_obj = Order.from_dict(order)
                orders_parsed.append(order_obj)
            except:

                return None, True

        return orders_parsed, False

    return locals()


def _simord_gpt(parser_ns):
    from types import SimpleNamespace
    def parser():
        return parser_ns

    MODEL = 'gpt-4o-2024-11-20'


    def decode(response):
        try:
            content = response['choices'][0]['message'].get('content')
            wrapped = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            orders, error = parser()['get_orders_from_model_response'](wrapped)
            if error:
                return [], False, 'order_parser_failure'
            return [o.to_dict() for o in orders], True, None
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            return [], False, 'order_parser_exception:' + type(exc).__name__

    return SimpleNamespace(**locals())


def _simord_mediphi_helpers():

    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


    # Reject truncated output before checking order fields and integer provenance references.
    def decode(response):
        if response['choices'][0].get('finish_reason') != 'stop':
            return [], False, 'non_stop_finish'
        text = (response['choices'][0]['message'].get('content') or '').strip().replace('```json', '').replace('```', '').strip()
        try:
            value = json.loads(text)
            if not isinstance(value, list) or any(not isinstance(o, dict) for o in value):
                raise ValueError('Expected an array of order objects')
            normalized = [{k: o.get(k) for k in ('description', 'order_type', 'reason', 'provenance')} for o in value]
            for o in normalized:
                if not isinstance(o['description'], str) or not isinstance(o['order_type'], str) or o['reason'] is not None and not isinstance(o['reason'], str):
                    raise ValueError('Unscorable order field')
                if o['provenance'] is not None and (not isinstance(o['provenance'], list) or any(type(x) is not int for x in o['provenance'])):
                    raise ValueError('Unscorable provenance')
            return normalized, True, None
        except (ValueError, TypeError) as exc:
            return [], False, str(exc)

    return SimpleNamespace(**locals())

def simord_api(transport, model):
    """Normalize SIMORD requests before calling the supplied transport."""
    def call(request,purpose):
        body=copy.deepcopy(request)
        body['model']=model
        body.pop('enable_thinking',None)
        return transport.call(body,purpose)
    result=SimpleNamespace(call=call)
    # Refusal state prevents retries of rejected source requests.
    for name in ('rejections',):
        if hasattr(transport,name):
            setattr(result,name,getattr(transport,name))
    return result


def parsed_array(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    for candidate in [text, text[text.find('['):text.rfind(']') + 1]]:
        try:
            x = json.loads(candidate)
            if isinstance(x, list): return True, x
        except (ValueError, TypeError): pass
    return False, []


# Expose the chat and embedding methods expected by the retrieval reader.
def synur_gpt52_client(api, TLS):
    import types

    def compact(obj):
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    def digest(obj):
        return sha(compact(obj).encode())

    def objectify(x):
        if isinstance(x, dict): return types.SimpleNamespace(**{k: objectify(v) for k, v in x.items()})
        if isinstance(x, list): return [objectify(v) for v in x]
        return x

    class Client:
        def __init__(self, api):
            self.api = api
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.chat_create))
            self.embeddings = types.SimpleNamespace(create=self.embed_create)

        def embed_create(self, **kw):
            return objectify(self.api.call(kw, 'synur/embeddings/' + digest(kw)))

        def chat_create(self, **kw):
            arm = TLS.arm; did = TLS.did
            review = kw['messages'][0]['content'].startswith('Return only')
            stage = 'audit' if review else 'reader'
            if arm == 'source_guided' and TLS.source_context:
                kw = copy.deepcopy(kw)
                kw['messages'][-1]['content'] += '\n\n' + TLS.source_context
            resp = self.api.call(kw, arm + '/' + did + '/' + stage)
            content = resp['choices'][0]['message'].get('content') or ''
            finish = resp['choices'][0].get('finish_reason')
            TLS.complete_array = TLS.complete_array or (finish == 'stop' and parsed_array(content)[0])
            forwarded = copy.deepcopy(resp)
            # Do not let a truncated response appear as a complete observation array.
            if finish != 'stop': forwarded['choices'][0]['message']['content'] = ''
            return objectify(forwarded)

    return Client(api)


# Normalize chat requests and track unfinished calls so partial results remain identifiable.
def synur_gpt52_api(transport, TLS):


    def compact(obj):
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    def digest(obj):
        return sha(compact(obj).encode())

    def call(request,purpose):
        payload = copy.deepcopy(request)
        if 'messages' in payload:
            route = '/chat/completions'; payload['model'] = 'gpt-5.2'
            payload.pop('enable_thinking', None)
            if 'max_tokens' in payload: payload['max_completion_tokens'] = payload.pop('max_tokens')
        else:
            route = '/embeddings'
        key = digest({'route': route, 'payload': payload})
        if not hasattr(TLS, 'pending_requests'): TLS.pending_requests = set()
        try:
            response=transport.call(payload,purpose)
        except Exception:
            TLS.pending_requests.add(key)
            raise
        TLS.pending_requests.discard(key)
        return response
    return SimpleNamespace(call=call)


# Translate thinking options only when requests are routed to a local model server.
def stupp_wire(profile):
    import prompts
    MODEL=prompts.STUPP[profile]["model"]
    LOCAL_MODEL=prompts.STUPP[profile]["local_model"]

    def wire_request(request, route, embedding=False):
        """Translate provider options for local model inference."""
        req = json.loads(json.dumps(request, ensure_ascii=False))
        if route == "local" and not embedding:
            if "enable_thinking" in req:
                req.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(req.pop("enable_thinking"))
            if "thinking_budget" in req:
                req.pop("thinking_budget")
            if req.get("model") == MODEL and LOCAL_MODEL != MODEL:
                req["model"] = LOCAL_MODEL
        return req

    return wire_request
