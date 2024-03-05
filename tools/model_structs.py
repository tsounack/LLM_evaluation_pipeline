from pydantic import BaseModel

class SymptomBinary(BaseModel):
    status: bool

class SymptomMultilabel(BaseModel):
    Anxiety: bool
    Concentration_Problems: bool
    Constipation: bool
    Cough: bool
    Diarrhea: bool
    Fatigue: bool
    Fever: bool
    Headache: bool
    Nausea: bool
    Numbness_and_Tingling: bool
    Pain: bool
    Poor_Appetite: bool
    Rash: bool
    Shortness_of_Breath: bool
    Trouble_Drinking_Fluids: bool
    Vomiting: bool
    Other: bool