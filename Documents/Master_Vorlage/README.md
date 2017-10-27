# zhaw-templates
This repo contains LaTeX templates for zhaw documents.

## zhawreprt.cls
This is the documentclass for zhaw documents. It formats the document and loads the needed packages. There are two package options you should know about:
- `german`/`ngerman` or `english` - switches some predefined texts to the passed language.
- `final` - ignores all the notes written in the file. If this flag is not passed on to the package, the gray colored notes will be displayed in the pdf.

It also provides the user with new or modified commands such as:
- `\maketitle` - modified title page using the zhaw standard.
- `\makedeclarationoforiginality` - prints the language specific declaration of originality.
- `\notes{...}` - print grey colored notes on the pdf. They disappear when the package option flag `final` was set.
- `\logofilename{...}` - sets the path of the desired logo. Browser the `./images/logos/` folder for all the possible logos.
- `\projecttype{...}` - either `PA`(project work) or `BA`(bachelor's thesis). Affects the title and the declaration of orginality.
- `\major{...}` - sets the name of the major.
- `\shorttitle{...}` - sets the short form of the title page to be used in the header.
- `\authors{...}` - sets the authors. Affects the title page.
- `\mainreferee{...}` - sets the main referee.
- `\referee{...}` - sets the referee.
- `\industrypartner{...}` - sets the industry partner.
- `\extreferee{...}` - sets the external referee.
- `\setdate{...}` - sets the date to be displayed.

## zhawDocument.tex
This is the document of interest. It basically holds all the documentation for the project in it. Parts that are not needed can easily be commented out.

## zhawSetup.tex
This is the setup for all the documents. It holds the information regarding authors, title, logo etc.

## zhawFunctionalSpecificationDocument.tex
This document provides basic commands to build up the functional specifications(FSD). There are additional commands:
- `\AddRequirement{...}` - formats a requirement.
- `\AddFRequirement{...}` - adds a functional requirement(F###).
- `\AddNonFRequirement{...}` - adds a non-functional requirement(R###).
- `\AddTestRequirement{...}` - adds a test requirement(T###).

## zhawGanttDiagram.tex
This here provides a template for a Gantt diagram.

## zhawNetworkPlan.tex
The network plan is the following step after the FSD. There are additional commands as well:
- `\NewMainNode{newNodeName}{position}{text}` - Very first node to be placed in a tree.
  - `newNodeName` - name to be used for the label of the node.
  - `position` - coordinates or coordinate label to place the node at.
  - `text` - text to be placed inside the node.
- `\AddMainNode{newNodeName}{parentNode}{xOffset}{yOffset}{text}` - Place a main node in relation to another node and connect with an arrow.
  - `newNodeName` - name to be used for the label of the node.
  - `parentNode` - label of the anchor node.
  - `xOffset` - offset of the new node on the x-axis.
  - `yOffset` - offset of the new node on the y-axis.
  - `text` - text to be placed inside the node.
- `\CreateNodeOffset{newLabel}{parentNode}` - creates an offset coordinate to let the arrow start on the center of the bottom.
  - `newLabel` - name to be used for the new coordinate.
  - `parentNode` - label of the anchor node.
- `\NewSubNode{newNodeName}{parentNode}{text}` - Make a subnode below a parent node.
  - `newNodeName` - name to be used for the label of the node.
  - `position` - coordinates or coordinate label to place the node at.
  - `text` - text to be placed inside the node.
- `\AddSubNode{newNodeName}{parentNode}{belowSubNode}{text}`- places a subnode below a previous subnode.
  - `newNodeName` - name to be used for the label of the node.
  - `position` - coordinates or coordinate label to place the node at.
  - `belowSubNode` - label of the previous subnode.
  - `text` - text to be placed inside the node.
