from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd
from langchain_core.messages import BaseMessage

class MainState(TypedDict):
    # 1. Interview Agent Output
    messages: List[BaseMessage]
    user_profile: Dict[str, Any] 
    
    # 2. NCS Agent Input/Output
    search_config: Dict[str, Any] # e.g. {'use_mmr': True, 'lambda_mult': 0.5}
    job_category_codes: List[int]
    recommendations: List[Dict]
    
    # 3. Posting Agent Output
    final_postings: Optional[pd.DataFrame]
    
    # 4. Post Manager Output
    selected_jobs: List[int]
    saved_jobs: List[Dict]
