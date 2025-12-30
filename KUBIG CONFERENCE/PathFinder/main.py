import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.state import MainState
from agents.interview_agent import run_interview_session
from agents.ncs_job_recommender_agent import run_ncs_agent
from agents.posting_recommender_agent import run_posting_recommender_agent
from agents.posting_manager_agent import PostManagerAgent

def main():
    print("============================================================")
    print("🚀 Unified Job Posting Recommendation Agent Started")
    print("============================================================")

    # 1. Initialize State
    state: MainState = {
        "messages": [],
        "user_profile": {},
        "search_config": {},
        "job_category_codes": [],
        "recommendations": [],
        "final_postings": None,
        "selected_jobs": [],
        "saved_jobs": []
    }

    # 2. Run Interview Agent
    print("\n[Step 1] Starting Interview Session...")
    user_profile = run_interview_session()
    state["user_profile"] = user_profile
    print("\n✅ Interview Completed.")
    
    # 3. Ask for Diversity Preference (MMR)
    print("\n[Step 2] Configuring Search Diversity...")
    while True:
        try:
            choice = input("\n직무 추천 시 얼마나 다양한 직무를 보고 싶으신가요?\n1: 비슷함 (정확도 중심)\n2: 보통\n3: 다양함 (새로운 발견 중심)\n선택 (1-3): ").strip()
            if choice == '1':
                lambda_mult = 0.3
                break
            elif choice == '2':
                lambda_mult = 0.5
                break
            elif choice == '3':
                lambda_mult = 0.7
                break
            else:
                print("⚠️ 1, 2, 3 중에서 선택해주세요.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return

    state["search_config"] = {"use_mmr": True, "lambda_mult": lambda_mult}
    print(f"✅ Search Configured: lambda_mult={lambda_mult}")

    # 4. Run NCS Job Agent
    print("\n[Step 3] Running NCS Job Recommendation Agent...")
    ncs_output = run_ncs_agent(state)
    state["job_category_codes"] = ncs_output.get("job_category_codes", [])
    state["recommendations"] = ncs_output.get("recommendations", [])
    print(f"✅ NCS Recommendations Generated: {len(state['recommendations'])} items")

    # 5. Run Posting Agent
    print("\n[Step 4] Running Job Posting Matching Agent...")
    posting_output = run_posting_recommender_agent(state)
    state["final_postings"] = posting_output.get("final_postings")
    
    if state["final_postings"] is None or state["final_postings"].empty:
        print("⚠️ No matching job postings found.")
    else:
        print(f"✅ Job Postings Matched: {len(state['final_postings'])} items")

    # 6. Run Post Manager Agent
    if state["final_postings"] is not None and not state["final_postings"].empty:
        print("\n[Step 5] Starting Post Manager (Save to Notion/Calendar)...")
        manager = PostManagerAgent()
        manager.start(state["final_postings"])
        
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                if not user_input.strip():
                    continue
                
                is_done = manager.chat(user_input)
                
                if is_done:
                    saved = manager.get_saved_jobs()
                    state["saved_jobs"] = saved
                    print(f"\n✨ {len(saved)} jobs saved!")
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break

    print("\n============================================================")
    print("🎉 All Steps Completed. Goodbye!")
    print("============================================================")

if __name__ == "__main__":
    main()
