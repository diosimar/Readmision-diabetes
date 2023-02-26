from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_null_prediction():
    response = client.post('/v1/prediction', json = {
                                                    
                                                    'age'                         :  0,
                                                    'num_procedures'              :  0,
                                                    'number_diagnoses'            :  0,
                                                    'number_emergency'            :  0,
                                                    'number_inpatient'            :  0,
                                                    'number_outpatient'           :  0,
                                                    'time_in_hospital'            :  0,
                                                    'visits_sum'                  :  0,
                                                    'number_medicaments_changes'  :  0,
                                                    'acarbose'                    :  0,
                                                    'acetohexamide'               :  0,
                                                    'chlorpropamide'              :  0,
                                                    'diabetesMed'                 :  0,
                                                    'glimepiride-pioglitazone'    :  0,
                                                    'glipizide-metformin'         :  0,
                                                    'metformin-pioglitazone'      :  0,
                                                    'metformin-rosiglitazone'     :  0,
                                                    'miglitol'                    :  0,
                                                    'tolbutamide'                 :  0,
                                                    'troglitazone'                :  0
                                                    })
    assert response.status_code == 200
    assert response.json()['Readmitted'] == 0

def test_random_prediction():
    response = client.post('/v1/prediction', json = {
                                                    'age'                         :  45,
                                                    'num_procedures'              :  2,
                                                    'number_diagnoses'            :  2,
                                                    'number_emergency'            :  2,
                                                    'number_inpatient'            :  2,
                                                    'number_outpatient'           :  2,
                                                    'time_in_hospital'            :  2,
                                                    'visits_sum'                  :  2,
                                                    'number_medicaments_changes'  :  2,
                                                    'acarbose'                    :  2,
                                                    'acetohexamide'               :  2,
                                                    'chlorpropamide'              :  2,
                                                    'diabetesMed'                 :  2,
                                                    'glimepiride-pioglitazone'    :  2,
                                                    'glipizide-metformin'         :  2,
                                                    'metformin-pioglitazone'      :  2,
                                                    'metformin-rosiglitazone'     :  2,
                                                    'miglitol'                    :  2,
                                                    'tolbutamide'                 :  2,
                                                    'troglitazone'                :  2
                                                })
    assert response.status_code == 200
    assert response.json()['Readmitted'] != 0 