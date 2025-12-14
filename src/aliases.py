SCHOOL_ALIASES = {
    "iit_bombay": ["IIT Bombay", "Indian Institute of Technology Bombay", "IITB", "IIT-Bombay"],
    "iit_delhi": ["IIT Delhi", "Indian Institute of Technology Delhi", "IITD", "IIT-Delhi"],
    "iit_madras": ["IIT Madras", "Indian Institute of Technology Madras", "IITM", "IIT-Madras"],
    "iit_kanpur": ["IIT Kanpur", "Indian Institute of Technology Kanpur", "IITK", "IIT-Kanpur"],
    "iit_kharagpur": ["IIT Kharagpur", "Indian Institute of Technology Kharagpur", "IIT-KGP", "IITKGP"],
    "iisc": ["IISc", "Indian Institute of Science", "IISc Bangalore", "IISc Bengaluru"],
    "bits_pilani": ["BITS Pilani", "Birla Institute of Technology and Science", "BITS"],
    "stanford": ["Stanford", "Stanford University", "Stanford GSB", "Stanford Graduate School of Business"],
    "mit": ["MIT", "Massachusetts Institute of Technology", "MIT Sloan"],
    "harvard": ["Harvard", "Harvard University", "Harvard Business School", "HBS"],
    "berkeley": ["UC Berkeley", "Berkeley", "University of California, Berkeley", "UCB", "Cal"],
    "cmu": ["CMU", "Carnegie Mellon", "Carnegie Mellon University"],
    "caltech": ["Caltech", "California Institute of Technology"],
    "princeton": ["Princeton", "Princeton University"],
    "yale": ["Yale", "Yale University"],
    "columbia": ["Columbia", "Columbia University"],
    "nyu": ["NYU", "New York University"],
    "upenn": ["UPenn", "Penn", "University of Pennsylvania", "Wharton"],
    "oxford": ["Oxford", "University of Oxford", "Oxford University"],
    "cambridge": ["Cambridge", "University of Cambridge", "Cambridge University"],
    "du": ["Delhi University", "DU", "University of Delhi"],
    "iim_ahmedabad": ["IIM Ahmedabad", "IIM-A", "IIMA", "Indian Institute of Management Ahmedabad"],
    "iim_bangalore": ["IIM Bangalore", "IIM-B", "IIMB", "Indian Institute of Management Bangalore"],
    "nit": ["NIT", "National Institute of Technology"],
    "vit": ["VIT", "Vellore Institute of Technology"],
    "srm": ["SRM", "SRM University", "SRM Institute of Science and Technology"],
}

LOCATION_ALIASES = {
    "bangalore": ["Bangalore", "Bengaluru", "Karnataka", "Blr", "BLR"],
    "mumbai": ["Mumbai", "Bombay", "Maharashtra"],
    "delhi": ["Delhi", "New Delhi", "NCR", "Gurgaon", "Gurugram", "Noida"],
    "hyderabad": ["Hyderabad", "Hyd", "Telangana", "Secunderabad"],
    "chennai": ["Chennai", "Madras", "Tamil Nadu"],
    "pune": ["Pune", "Poona", "Maharashtra"],
    "kolkata": ["Kolkata", "Calcutta", "West Bengal"],
    "san_francisco": ["San Francisco", "SF", "Bay Area", "Silicon Valley"],
    "new_york": ["New York", "NYC", "New York City", "Manhattan", "Brooklyn"],
    "seattle": ["Seattle", "Washington", "WA"],
    "austin": ["Austin", "Texas", "TX"],
    "boston": ["Boston", "Massachusetts", "MA"],
    "london": ["London", "UK", "United Kingdom"],
    "singapore": ["Singapore", "SG"],
    "dubai": ["Dubai", "UAE", "United Arab Emirates"],
}

SKILL_ALIASES = {
    "frontend": ["frontend", "front-end", "react", "reactjs", "vue", "vuejs", "angular", 
                 "javascript", "typescript", "ui engineer", "ui developer", "web developer", 
                 "nextjs", "html", "css", "tailwind"],
    "backend": ["backend", "back-end", "node", "nodejs", "django", "flask", "fastapi", 
                "spring boot", "java", "python", "golang", "api development", "server-side"],
    "fullstack": ["fullstack", "full stack", "full-stack", "mern", "mean"],
    "machine_learning": ["machine learning", "ml", "deep learning", "neural networks", 
                         "tensorflow", "pytorch", "nlp", "natural language processing", 
                         "computer vision", "ai", "artificial intelligence", "data science"],
    "data_science": ["data science", "data scientist", "analytics", "pandas", "numpy", 
                     "statistics", "data analysis", "data analyst"],
    "devops": ["devops", "docker", "kubernetes", "k8s", "ci/cd", "aws", "cloud", 
               "infrastructure", "sre", "site reliability"],
    "product": ["product manager", "product management", "pm", "product owner", "product lead"],
    "design": ["designer", "ui/ux", "ux designer", "ui designer", "product designer", "figma"],
    "mobile": ["mobile", "ios", "android", "react native", "flutter", "swift", "kotlin"],
    "security": ["security", "cybersecurity", "infosec", "penetration testing", "security engineer"],
}

COMPANY_ALIASES = {
    "google": ["Google", "Alphabet", "Google Cloud", "GCP", "YouTube", "DeepMind"],
    "meta": ["Meta", "Facebook", "Instagram", "WhatsApp", "Oculus"],
    "amazon": ["Amazon", "AWS", "Amazon Web Services", "Twitch", "Whole Foods"],
    "microsoft": ["Microsoft", "Azure", "LinkedIn", "GitHub", "Xbox"],
    "apple": ["Apple", "Apple Inc"],
    "netflix": ["Netflix"],
    "faang": ["Google", "Meta", "Facebook", "Amazon", "Apple", "Netflix", "Microsoft"],
    "startup": ["startup", "founder", "co-founder", "cofounder", "entrepreneur", "ceo"],
}


def get_canonical_school(school_name: str) -> str:
    school_lower = school_name.lower()
    for canonical, variations in SCHOOL_ALIASES.items():
        for var in variations:
            if var.lower() in school_lower or school_lower in var.lower():
                return canonical
    return school_lower.replace(" ", "_")[:20]


def get_school_variations(school_name: str) -> list:
    school_lower = school_name.lower()
    for canonical, variations in SCHOOL_ALIASES.items():
        for var in variations:
            if var.lower() in school_lower or school_lower in var.lower():
                return variations
    return [school_name]


def expand_location(location: str) -> list:
    loc_lower = location.lower()
    for canonical, variations in LOCATION_ALIASES.items():
        for var in variations:
            if var.lower() == loc_lower or loc_lower in var.lower():
                return variations
    return [location]


def expand_skill(skill: str) -> list:
    skill_lower = skill.lower()
    for category, skills in SKILL_ALIASES.items():
        if skill_lower in [s.lower() for s in skills]:
            return skills
    return [skill]


def get_alias_context_for_prompt() -> str:
    lines = ["## Quick Reference (use these, expand further as needed):"]
    
    lines.append("\nSCHOOLS:")
    for canonical, variations in list(SCHOOL_ALIASES.items())[:10]:
        lines.append(f"  {variations[0]} = {', '.join(variations[1:3])}")
    
    lines.append("\nLOCATIONS:")
    for canonical, variations in list(LOCATION_ALIASES.items())[:8]:
        lines.append(f"  {variations[0]} = {', '.join(variations[1:3])}")
    
    lines.append("\nSKILLS:")
    for category, skills in list(SKILL_ALIASES.items())[:5]:
        lines.append(f"  {category} = {', '.join(skills[:5])}")
    
    return "\n".join(lines)