/*
 * xtc_tree.hh
 *
 *  Created on: Feb 14, 2020
 *      Author: tonglin
 */



#ifndef XTC_TREE_HH_
#define XTC_TREE_HH_

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "xtc_io_api_c.h"

using namespace std;

typedef struct XTC_Tree_Node xtc_node;
struct XTC_Tree_Node{
    xtc_object* my_obj;//all xtc obj related info stored here

    //------------- Tree related info ----------------
    vector<string> path_abs; //tokenized path
    xtc_node* parent;//NULL for root
    vector<xtc_node*> children;
    int is_virtual;
};

xtc_node* new_xtc_node(xtc_object* obj);
xtc_node* new_test_node(const char* path);
int add_xtc_node(xtc_node* root, xtc_node* new_node);
xtc_node* find_xtc_node(xtc_node* root, const char* path);
xtc_object** get_children_obj(xtc_object* group, int* num_out);
int print_tree_node(xtc_node* n);
int print_tree(xtc_node* root);

/*
 * add node
 *
 * */

#endif /* XTC_TREE_HH_ */
