import requests
import json


def createApplicationEntity(AEname, serverURL, role):
    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
        "Content-Type":"application/json;ty=2"
    }
    body = {
        "m2m:ae":{
            "rn":AEname,
            "api":"ID_api_raleeshinjo",
            "lbl":["key1","key2"],
            "rr": True,
            "poa":["MQTT|"]
        }
    }
    res = requests.post(serverURL+"/Mobius", headers=header, json=body)
    print(res.json())


def createContainer(AEname, dir, serverURL, role, acpi):
    dir = addSlash(dir)
    header = {
        "Accept": "application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
        "Content-Type":"application/vnd.onem2m-res+json; ty=3"
    }
    name = dir.split('/')[-1]
    udir = '/'.join(dir.split('/')[0:-1])
    print("udir:", udir)
    print("name:", name)
    body = {
        "m2m:cnt":{
            "rn":name,
            "lbl":[name],
            "mbs":16384,
            "acpi":["Mobius/"+acpi]
        }
    }
    res = requests.post(serverURL+"/Mobius/"+AEname+udir, headers=header, json=body)
    print(res.json())


def getContainer(AEname, dir, serverURL, role):
    dir = addSlash(dir)
    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
    }
    res = requests.get(serverURL+"/Mobius/"+AEname+dir, headers=header, data="")
    print(res.json())


def deleteContainer(AEname, dir, serverURL, role):
    dir = addSlash(dir)
    header = {
        "Accept": "application/json",
        "locale":"ko",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role
    }
    res = requests.delete(serverURL+"/Mobius/"+AEname+dir, headers=header, data = "")
    print(res.json())


#have to make COMMAND
def createSubscription(AEname, dir, webserver, serverURL, role):
    dir = addSlash(dir)
    AEname = addSlash(AEname)
    header = {
        "X-M2M-Origin": role,
        "X-M2M-RI": "3408",
        "Content-Type": "application/json;ty=23"
    }
    body = {
        "m2m:sub": {
            "rn": "sub",
            "nu": [webserver + "?ct=json"],
            "nct": 1,
            "enc": {
                "net": [3]
            }
        }
    }
    res = requests.post(serverURL+"/Mobius"+AEname+dir, headers=header, json=body)
    print(res.json())


def createContentInstance(AEname, dir, con, serverURL, role):
    dir = addSlash(dir)
    header = {
        "Accept": "application/json",
        "X-M2M-Origin": role,
        "X-M2M-RI": "3408",
        "Content-Type": "application/json;ty=4"
    }
    body = {
        "m2m:cin":{
            "con": con
        }
    }
    res = requests.post(serverURL+"/Mobius/"+AEname+dir, headers=header, json=body)
    print(res.json())


#get ae_list
def getApplicationEntityList(serverURL, role): # return AE name list
    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
    }
    res = requests.get(serverURL+"/Mobius?fu=1&ty=2&lim=20", headers=header, data="")

    #print(res.json())
    
    res = res.json()['m2m:uril']
    for i, s in enumerate(res): # make ae name list
        res[i] = s[7:]
    
    return res


def getOnlyIDUserAEs(serverURL, role): # return AE name list
    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
    }
    res = requests.get(serverURL+"/Mobius?fu=1&ty=2&lim=9999", headers=header, data="")
    res = res.json()['m2m:uril']

    ae_list = [] 
    for i, s in enumerate(res): # make ae name list
        res[i] = s[7:]
        if("ID_USER_" in res[i]):
            ae_list.append(res[i])
    return ae_list


#remove all_ae
def removeAllApplicationEntity(serverURL, role):
    ae_list = getApplicationEntityList(serverURL, role)
    print(ae_list)
    for step in ae_list:
        removeApplicationEntity(serverURL, step, role)


#get ae_ri
def getApplicationEntityRI(serverURL, AEname, role):
    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
    }
    res = requests.get(serverURL+"/Mobius/"+AEname, headers=header, data="")
    res = res.json()
    print(res)
    ae_ri = res["m2m:ae"]["ri"]
    return ae_ri


#remove AEname
def removeApplicationEntity(serverURL, AEname, role):
    ri = getApplicationEntityRI(serverURL, AEname, role)

    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":ri,
    }
    res = requests.delete(serverURL+"/Mobius/"+AEname, headers=header, data="")
    res = res.json()
    print(res)


def createAccessControlPolicy(serverURL, ACPname, doctor, manager, nurse, user, gateway, role):
    header = {
        "Accept":"application/json",
        "X-M2M-RI":"3408",
        "X-M2M-Origin":role,
        "Content-Type":"application/vnd.onem2m-res+json; ty=1",
    }
    body = {
        "m2m:acp" : {
            "rn" : ACPname,
            "pv" : {
                "acr" : [{
                "acco" : [],
                    "acor" : [doctor],
                    "acop" : "55"
                }, 
                {
                    "acor" : [manager],
                    "acop" : "61"
                },
                {
                    "acor" : [nurse, user],
                    "acop" : "50"
                },
                {
                    "acor" : [gateway],
                    "acop" : "17"
                }]
            },
            "pvs" : {
                "acr" : [{
                "acco" : [],
                    "acor" : [manager],
                    "acop" : "63"
                }]
            }
        }
    }
    res = requests.post(serverURL+"/Mobius", headers=header, json=body)
    print(res.json())


def addSlash(string):
    if string == "":
        return string
    if string[0] != '/':
        string = '/' + string
    return string
