
//todo create EKS using terraform 
//todo create the eks cluster
// todo iam roles for cluster
// todo iam policy for the cluster
//todo create security group for the cluster
//create RBAC 
// hardning cluster 

resource "aws_iam_role" "name" {
 name = var.cluster-name 
 assume_role_policy = <<POLICY
 {
     "Version" : "2017-10-17",
     "Statement" : [
         "Effect": "Allow"
     ]
 }
 POLICY
}
resource "aws_iam_role_policy_attachment" "name" {
    policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
    role = aws_iam_role.name.name
  
}
resource "aws_security_group" "name" {
  name   = "eks-security-gp-01"
  vpc_id = var.vpc-id
  egress {
    from_port   = 0
    protocol    = "-1"
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_eks_cluster" "cluster" {
  name     = "clustter-eks-rbac"
  role_arn = aws_iam_role.name.arn
  vpc_config {
    subnet_ids = [aws_security_group.name.id]
  }
}