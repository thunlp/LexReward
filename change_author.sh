#!/bin/bash
cd /home/dx/LexReward-main

# 配置你的信息
git config user.name "Doradai28"
git config user.email "3543631320@qq.com"

# 修改最近3个提交的作者
git filter-branch -f --env-filter '
CORRECT_NAME="Doradai28"
CORRECT_EMAIL="3543631320@qq.com"

export GIT_COMMITTER_NAME="$CORRECT_NAME"
export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
export GIT_AUTHOR_NAME="$CORRECT_NAME"
export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
' HEAD~3..HEAD

echo "作者信息已修改完成"
