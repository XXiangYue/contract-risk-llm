"""
合同合规审查API服务
基于vLLM的OpenAI兼容API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import requests
import os

app = FastAPI(title="合同合规审查API", version="1.0.0")

# vLLM服务配置
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# 合同审查系统提示
SYSTEM_PROMPT = """你是一个专业的企业合同合规审查专家。你的职责是：
1. 审查合同中的潜在风险条款
2. 识别不公平条款（如霸王条款）
3. 检查合规性问题
4. 提供修改建议

请仔细阅读合同内容，并按照以下格式输出审查结果：
- 风险等级：高/中/低
- 风险类型：违约金/管辖权/违约条款/保密义务/其他
- 具体条款：摘录相关条款
- 风险说明：说明风险点
- 修改建议：提供具体的修改建议"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen3.5-2B-contract"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 0.9


class ContractReviewRequest(BaseModel):
    contract_text: str
    review_type: Optional[str] = "full"  # full/risk/compliance


@app.get("/")
async def root():
    return {"message": "合同合规审查API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """健康检查"""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        return {"status": "healthy", "vllm": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI兼容的聊天接口"""
    try:
        # 转换为vLLM格式
        payload = {
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        response = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contract/review")
async def contract_review(request: ContractReviewRequest):
    """合同审查专用接口"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请审查以下合同：\n\n{request.contract_text}"}
        ]
        
        payload = {
            "model": "Qwen/Qwen3.5-2B-contract",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        
        response = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        return {
            "review_result": result["choices"][0]["message"]["content"],
            "review_type": request.review_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contract/batch-review")
async def batch_review(contracts: List[str]):
    """批量合同审查接口"""
    results = []
    
    for contract in contracts:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请审查以下合同：\n\n{contract}"}
            ]
            
            payload = {
                "model": "Qwen/Qwen3.5-2B-contract",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2048,
            }
            
            response = requests.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    "status": "success",
                    "review": result["choices"][0]["message"]["content"]
                })
            else:
                results.append({
                    "status": "error",
                    "error": response.text
                })
        
        except Exception as e:
            results.append({
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
