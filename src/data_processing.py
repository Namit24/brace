import json
from typing import Dict, List, Any, Optional
def load_json(filepath: str) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class ActorProcessor:
    def get_actor_id(self, actor: Dict) -> str:
        identities = actor.get("platform_identities", [])
        if identities:
            return identities[0].get("platform_id", "unknown")
        return "unknown"
    
    def extract_education_chunks(self, actor: Dict) -> List[Dict[str, Any]]:
        actor_id = self.get_actor_id(actor)
        name = actor.get("profile", {}).get("name", "Unknown")
        education_list = actor.get("professional", {}).get("education", [])
        
        chunks = []
        for edu in education_list:
            school = edu.get("school", "")
            if not school or school.strip() in ["*", ""]:
                continue
                
            degree = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            
            edu_text = f"{school}"
            if degree:
                edu_text += f", {degree}"
            if field:
                edu_text += f" in {field}"
            
            chunks.append({
                "actor_id": actor_id,
                "name": name,
                "chunk_type": "education",
                "text": edu_text,
                "school": school,
                "degree": degree,
                "field_of_study": field,
            })
        
        return chunks
    
    def extract_skills_chunk(self, actor: Dict) -> Optional[Dict[str, Any]]:
        actor_id = self.get_actor_id(actor)
        name = actor.get("profile", {}).get("name", "Unknown")
        profile = actor.get("profile", {})
        professional = actor.get("professional", {})
        
        headline = profile.get("headline", "")
        bio = profile.get("bio", "")
        
        job_titles = []
        job_descriptions = []
        for exp in professional.get("work_experience", []):
            title = exp.get("title", "")
            if title:
                job_titles.append(title)
            desc = exp.get("description", "")
            if desc:
                job_descriptions.append(desc[:500])
        
        skills_text = f"Skills and expertise: {headline}. "
        skills_text += f"Roles: {', '.join(job_titles[:5])}. "
        if bio:
            skills_text += f"Background: {bio[:300]}"
        
        return {
            "actor_id": actor_id,
            "name": name,
            "chunk_type": "skills",
            "text": skills_text,
            "job_titles": job_titles[:5],
        }
    
    def extract_companies_chunk(self, actor: Dict) -> Optional[Dict[str, Any]]:
        actor_id = self.get_actor_id(actor)
        name = actor.get("profile", {}).get("name", "Unknown")
        professional = actor.get("professional", {})
        
        companies = []
        roles = []
        for exp in professional.get("work_experience", []):
            company = exp.get("company_name", "")
            if company:
                companies.append(company)
            title = exp.get("title", "")
            if title:
                roles.append(f"{title} at {company}")
        
        if not companies:
            return None
        
        companies_text = f"Work experience at: {', '.join(companies)}. "
        companies_text += f"Roles: {', '.join(roles[:5])}"
        
        return {
            "actor_id": actor_id,
            "name": name,
            "chunk_type": "companies",
            "text": companies_text,
            "companies": list(set(companies)),
            "roles": roles[:5],
        }
    
    def extract_location_chunk(self, actor: Dict) -> Optional[Dict[str, Any]]:
        actor_id = self.get_actor_id(actor)
        name = actor.get("profile", {}).get("name", "Unknown")
        location = actor.get("profile", {}).get("location", "")
        
        if not location:
            return None
        
        location_text = f"Located in: {location}"
        
        return {
            "actor_id": actor_id,
            "name": name,
            "chunk_type": "location",
            "text": location_text,
            "location": location,
        }
    
    def get_full_profile(self, actor: Dict) -> Dict[str, Any]:
        actor_id = self.get_actor_id(actor)
        profile = actor.get("profile", {})
        professional = actor.get("professional", {})
        
        education = []
        for edu in professional.get("education", []):
            school = edu.get("school", "")
            if school and school.strip() not in ["*", ""]:
                education.append(school)
        
        companies = []
        for exp in professional.get("work_experience", []):
            company = exp.get("company_name", "")
            if company:
                companies.append(company)
        
        current = professional.get("current_position", {})
        current_role = ""
        if current:
            current_role = f"{current.get('title', '')} at {current.get('company', '')}"
        
        return {
            "actor_id": actor_id,
            "name": profile.get("name", "Unknown"),
            "headline": profile.get("headline", ""),
            "location": profile.get("location", ""),
            "bio": profile.get("bio", ""),
            "education": education,
            "companies": list(set(companies)),
            "current_role": current_role,
        }
    
    def process_actor(self, actor: Dict) -> Dict[str, Any]:
        return {
            "actor_id": self.get_actor_id(actor),
            "profile": self.get_full_profile(actor),
            "education_chunks": self.extract_education_chunks(actor),
            "skills_chunk": self.extract_skills_chunk(actor),
            "companies_chunk": self.extract_companies_chunk(actor),
            "location_chunk": self.extract_location_chunk(actor),
        }
    
    def process_all_actors(self, actors: List[Dict]) -> List[Dict[str, Any]]:
        """Process all actors."""
        return [self.process_actor(actor) for actor in actors]