/*
 * xtc_tree.cc
 *
 *  Created on: Feb 14, 2020
 *      Author: tonglin
 */
#include <iostream>
#include "util_debug.h"
#include "xtc_io_api_cpp.hh"
#include "xtc_tree.hh"
using namespace std;

xtc_node* new_xtc_node(xtc_object* obj){
    assert(obj);
    assert(obj->obj_path_abs);
    xtc_node* node = (xtc_node*)calloc(1, sizeof(xtc_node));
    node->my_obj = obj;
    node->path_abs = str_tok(obj->obj_path_abs, "/");
    if(node->path_abs.size() == 0)
        node->parent = NULL;
    return node;
}

//make virtual node for xtc tree node that doesn't have a real xtc obj, only serves as a directory node
xtc_node* new_virtual_xtc_node(int fd, vector<string> path_abs){
    assert(path_abs.size() > 0);
    printf("New virtual node path len = %d, ", path_abs.size());
    print_path(path_abs);
    printf("\n");
    DEBUG_PRINT
    xtc_object* vobj = (xtc_object*)calloc(1, sizeof(xtc_object));
    vobj->fd = fd;
    vobj->obj_path_abs = strdup(tok_to_str(path_abs).c_str());
    vobj->obj_type = XTC_GROUP;
    xtc_token_new((xtc_token_t*)(vobj->obj_token), 16);
    DEBUG_PRINT
    xtc_node* node = (xtc_node*)calloc(1, sizeof(xtc_node));
    node->my_obj = vobj;
    node->path_abs = path_abs;
    DEBUG_PRINT
    return node;
}

int add_xtc_node(xtc_node* root, xtc_node* new_node){
    assert(root && new_node);
    printf("new_node->path_abs.size() = %d\n", new_node->path_abs.size());
    if(new_node->path_abs.size() < 2){
        if(new_node->path_abs.size() == 0){
            printf("new_node path len = 0, invalid node.\n");
            return -1;
        }
        if(new_node->path_abs.size() == 1){
            DEBUG_PRINT
            for(auto i : root->children){
                if(!i->path_abs[0].compare(new_node->path_abs[0])){
                    printf("new_node has a conflicting path: %s\n", new_node->path_abs[0].c_str());
                    return -1;
                }
            }
            new_node->parent = root;
            root->children.push_back(new_node);
            return 0;
        }
    }

    xtc_node* insert_to = root;
    int index_n = 0;
    bool token_match = false;
    while(index_n <= new_node->path_abs.size() - 1){
        DEBUG_PRINT
        printf("insert_to path = %s\n", tok_to_str(insert_to->path_abs).c_str());

        if(index_n < new_node->path_abs.size() - 1){
            if(insert_to->children.size() > 0){
                token_match = false;
                for(auto i : insert_to->children){
                    DEBUG_PRINT
                    cout<<"New node path = "<< tok_to_str(new_node->path_abs) <<", new node token = "
                        <<new_node->path_abs[index_n] << ", child path = " << tok_to_str(i->path_abs)
                        << ", index_n = " << index_n
                        << endl;
                    if(0 == new_node->path_abs[index_n].compare(i->path_abs.back())){
                        printf("token match: token = %s, index_n = %d\n", i->path_abs.back().c_str(), index_n);
                        insert_to = i;
                        index_n++;
                        token_match = true;
                        break; //for
                    }  //else continue scan children
                }
            } else {//empty dir
                token_match = false;
            }

            if(!token_match){//need to add a virtual node as new group
                vector<string> new_path = insert_to->path_abs;
                new_path.push_back(new_node->path_abs[index_n]);
                xtc_node* new_dir_node = new_virtual_xtc_node(root->my_obj->fd, new_path);
                printf("Insert new_dir_node (%s) to %s\n", tok_to_str(new_path).c_str(), tok_to_str(insert_to->path_abs).c_str());
                new_dir_node->parent = insert_to;
                insert_to->children.push_back(new_dir_node);
                insert_to = new_dir_node;
                index_n++;
            }
        } else {// == size - 1: last one
            printf("Index_n = %d, reached last new_node token (%s), insert_to = %s, has %d children.\n", index_n, new_node->path_abs[index_n].c_str(),
                    tok_to_str(insert_to->path_abs).c_str(), insert_to->children.size());
            if(insert_to->children.size() > 0){
                for(auto i:insert_to->children){
                    if(new_node->path_abs[index_n].compare(i->path_abs[index_n])==0){
                        printf("End token match: invalid insert, return.\n");
                        return -1;
                    }// continue scaning
                }//no match, need to insert.
                break;//while
            } else {// do insert to a empty dir.
                break;//while
            }
        }
    }//while
    //insert actual node now
    new_node->parent = insert_to;
    printf("Insert actual node to : %s\n", tok_to_str(insert_to->path_abs).c_str());
    insert_to->children.push_back(new_node);

    DEBUG_PRINT

    return 0;
}

int print_tree_node(xtc_node* n){
    if(n->my_obj->obj_type != XTC_GROUP && n->my_obj->obj_type != XTC_ROOT_GROUP){
        //printf("Can not traverse non-group node, return.\n");
        return 0;
    }
    printf("node path = %s, path len = %d, ", tok_to_str(n->path_abs).c_str(), n->path_abs.size());//tok_to_str(n->path_abs).c_str(), n->my_obj->obj_path_abs


    if(n->parent)
        printf("parent path = %s, %d children:\n", n->parent->my_obj->obj_path_abs, n->children.size());//tok_to_str(n->parent->path_abs).c_str()
    else
        printf("%d children:\n", n->children.size());
    for(auto i:n->children){
        printf("        child path = %s, path len = %d\n", i->my_obj->obj_path_abs, i->path_abs.size());//tok_to_str(i->path_abs).c_str()
    }
    printf("---------------\n");
    return 0;
}

int print_tree(xtc_node* root){
    assert(root);
    print_tree_node(root);
    //printf("===============================\n");
    for(auto i:root->children){
        if(i)
            print_tree(i);
    }
    return 0;
}

xtc_node* new_test_node(char* path) {
    xtc_object* obj = (xtc_object*) calloc(1, sizeof(xtc_object));
    obj->fd = 5;

    obj->obj_path_abs = strdup(path);
    obj->obj_type = XTC_GROUP;
    xtc_node* node = new_xtc_node(obj);
    return node;
}
