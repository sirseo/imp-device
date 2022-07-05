From ubuntu:18.04

WORKDIR /app/

RUN apt-get update 
RUN apt-get install git -y
RUN apt-get install npm -y

COPY ./Mobius/package*.json ./ 
RUN npm install
COPY ./Mobius/ .


ENTRYPOINT ["node", "mobius.js"]

EXPOSE 7579

