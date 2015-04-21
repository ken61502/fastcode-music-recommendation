./spark-ec2 --key-pair=spark-mllib --identity-file=spark-mllib.pem -s 1 --region=us-west-1 --zone=us-west-1a --instance-type=m3.medium --copy-aws-credentials launch ml-cluster
ssh -i spark-mllib.pem root@ec2-54-153-22-174.us-west-1.compute.amazonaws.com
ln -s /etc/httpd/modules/libphp-5.6.so /etc/httpd/modules/libphp-5.5.so
service httpd restart
scp -i spark-mllib.pem -r svd_imf_spark.py root@ec2-52-8-41-165.us-west-1.compute.amazonaws.com:/spark/

./bin/spark-submit --master spark://ec2-52-8-41-165.us-west-1.compute.amazonaws.com:7077 linearReg-spark.py
./spark-ec2 destroy ml-cluster