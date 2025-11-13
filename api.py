from fastapi import APIRouter, BackgroundTasks
from model import API_Response_Model, Request_Model, Feedback_Model, Feedback_Response_Model
import asyncio
from service import HybridPersonalizedModelPG


class API():
    def __init__(self):
        self.router = APIRouter()
        self.register_routes()
        self.hybrid = HybridPersonalizedModelPG()

    async def run_process_task(self, records, categories, user_id):
        # try:
            print(f"Background processing Started")
            # await asyncio.sleep(5)
            result = await self.hybrid.process(records, categories, user_id)
            print("Response:\n",result)

        # except Exception as e:
        #     print("Background task error:", e)

    async def run_feedback_task(self, user_id):
        # try:
            print(f"Background processing Started")
            # await asyncio.sleep(5)
            result = await self.hybrid.feedback(user_id)
            print("Response:\n",result)

        # except Exception as e:
        #     print("Background task error:", e)

    def register_routes(self):
        @self.router.post("/process", response_model=API_Response_Model)
        async def process(req: Request_Model, background_tasks: BackgroundTasks) -> API_Response_Model:
            response = API_Response_Model()

            try:
                result = await self.hybrid.main(req.url, req.user_id)

                if result:
                    background_tasks.add_task(self.run_process_task, result, req.categories, req.user_id)
                    response.success = True
                    response.message = "Background task started"
                    response.error = None
                    response.data = "Processing..."
                    response.user_id = req.user_id

                else:
                    response.success = False
                    response.data = "Invalid URL or null user_id"
                    response.user_id = req.user_id

            except Exception as e:
                response.success = False
                response.message = str(e)

            return response

        @self.router.post("/feedback", response_model=dict)
        async def feedback(req: Feedback_Model, background_tasks: BackgroundTasks):
            try:
                result = await self.hybrid.feedback_signal(req.user_id)
                print(result)

                if result == req.user_id:
                    background_tasks.add_task(self.run_feedback_task, req.user_id)
                    return {
                        "success": True,
                        "error": None,
                        "user_id": req.user_id
                    }

                elif result == f"No records found against id: '{req.user_id}'":
                    return {
                        "success": False,
                        "data": result,
                        "error": None,
                        "user_id": req.user_id
                    }
                
                else:
                    return {
                    "success": False,
                    "error": "user_id cannot be null",
                    "user_id": "null"
                }

            except Exception as e:
                return {"error": e}
