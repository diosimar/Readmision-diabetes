from pydantic import BaseModel

class PredictionRequest(BaseModel):
    
    age                         :  float
    num_procedures              :  float
    number_diagnoses            :  float
    number_emergency            :  float
    number_inpatient            :  float
    number_outpatient           :  float
    time_in_hospital            :  float
    visits_sum                  :  float
    number_medicaments_changes  :  float
    acarbose                    :  float
    acetohexamide               :  float
    chlorpropamide              :  float
    diabetesMed                 :  float
    glimepiride-pioglitazone    :  float
    glipizide-metformin         :  float
    metformin-pioglitazone      :  float
    metformin-rosiglitazone     :  float
    miglitol                    :  float
    tolbutamide                 :  float
    troglitazone                :  float
     


class PredictionResponse(BaseModel):
    Readmitted                  :  int

   