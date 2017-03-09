import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders
import os
import datetime


 
def sendMail(to, fro, subject, text, files=[],server="localhost"):
    assert type(to)==list
    assert type(files)==list
 
 
    msg = MIMEMultipart()
    msg['From'] = fro
    msg['To'] = COMMASPACE.join(to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
 
    msg.attach( MIMEText(text) )
 
    for file in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(file,"rb").read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"'
                       % os.path.basename(file))
        msg.attach(part)
 
    smtp = smtplib.SMTP(server)
    smtp.sendmail(fro, to, msg.as_string() )
    smtp.close()
 
# Example:
#sendMail(['maSnun <bdmckean@gmail.com>'],'phpGeek <masnun@leevio.com>','Hello Python!','Heya buddy! Sy hello to Python! :)',['masnun.py','masnun.php'])
sendMail(['brian <bdmckean@gmail.com>'],'btrakdb <root@ec2-52-35-17-187.us-west-2.compute.amazonaws.com>','Daily Trial Report - {}!'.format(datetime.datetime.now().date()),
        'Daily report Biotrak Health v1.1 Database',
        ['Trial_model_report.csv',
            'Trial_report.csv',
                'Trial_sessions_report.csv',
                    'btrak_trial_classify.docx'])

